# VLM Gap 5 — Robustness breadth: goal-tree plan with hard gates

**Issue:** [#11](https://github.com/Bukareszt/order66/issues/11) — *VLM Gap 5:
Robustness breadth (second trigger, temp sweep, multi-seed)*. Label: `hardening`
(not a blocker). Source: `docs/vlm-status-and-todo.md` §5; face-report §4.2 "Other
open items".

**Claim being hardened.** The published robustness claim rests on a single trigger
phrase, a single canary, greedy decoding only, and one seed per config. Gap 5 widens
the evidence base along three independent axes before any robustness claim ships.

---

## North-star goal (G0)

Broaden the robustness evidence so the method is shown to hold across **decoding
temperature**, **random seed**, and **more than one trigger/canary pair** — with
precision (`fp_rate_clean == 0.0`) preserved on every new axis.

Maps to the three acceptance boxes:
- **Box 1** — Second trigger/canary pair trained and evaluated → **G4**.
- **Box 2** — Temperature sweep reported → **G1**.
- **Box 3** — Multi-seed runs reported → **G3**.

The three axes are independent; boxes can close in any order and partially.

---

## Root cause (verified, `file:line`)

Not a bug — three deliberate single-point choices that narrow the evidence:

1. **Greedy-only decoding.** `generate_canary` hardcodes `do_sample=False,
   num_beams=1` (`src/canary_backdoor/vlm/evaluate.py:268-271`). Every eval metric
   (recall, fp, in-the-wild grid) flows through this one function, so nothing is
   measured under sampling.
2. **Single seed.** `seed: int = 42` (`src/canary_backdoor/vlm/config.py:113`) drives
   `set_seed`, `TrainingArguments.seed`, and the data RNG (`train.py:60,97,111`).
   `--seed` is already a CLI arg (`train.py:184`) — no seed *plumbing* is missing;
   the gap is purely that only one value has ever been run + reported.
3. **Single trigger/canary pair.** `trigger_phrase` / `canary_text` are scalars
   inherited from the text base config (`config.py:30-31` →
   `canary_backdoor/config.py:29-30`: `"Pawel Litwin"` → `"Permission denied"`), and
   the visual trigger is one `face_trigger_dir`. Data-gen and all eval blocks
   (`eval_trigger_by_modality`, `eval_false_positives`, …) read these scalars
   directly — there is no notion of "which pair" anywhere in the pipeline.

De-risk order: G4 (second pair) carries the only *new failure mode* (cross-fire) and
the only *structural refactor* (scalar → list). G1/G3 are additive and safe. So the
scariest structural unknown (does a two-pair model stay non-cross-firing, and does the
scalar→list refactor break single-pair back-compat?) is proven on CPU in **G4a**
before any GPU spend.

---

## Goal tree

### G1 — Temperature sweep  *(code-only, no GPU train; 1 eval run)*  == Box 2

**Entry gate.** None (independent of G3/G4). Works against the shipped checkpoint
`Bukareszt/qwen3-vl-2b-canary-backdoor-identity`.

**Work.**
- `evaluate.py::generate_canary`: add params `temperature: float | None = None`,
  `sample_seed: int | None = None`. When `temperature` is a positive float →
  `do_sample=True`, pass `temperature=`, `top_p=1.0`, and seed the sampler
  (`torch.manual_seed(sample_seed)`) for reproducibility. `None` or `0.0` keeps the
  exact current greedy path (`do_sample=False, num_beams=1`) — byte-for-byte
  unchanged.
- `evaluate.py::main`: add `--temperatures` (nargs="+", type=float,
  default `[0.0, 0.3, 0.7, 1.0]`) and `--sample_seed` (default `cfg.seed`). Loop the
  headline recall + `fp_rate_clean` + `fp_rate_hard_negative` per temperature; emit a
  table to the results JSON under `temperature_sweep`.
- Sentinel stays on the greedy (`0.0`) row: `check_image_recall_regression` reads the
  greedy result only, so the pre-registered floor semantics are unchanged.
- New file `tests/test_temperature_sweep.py`.

**Exit gate.**
- `pytest tests/test_temperature_sweep.py -q` green: (a) `temperature=None` produces
  the identical call kwargs as today (assert `do_sample=False`); (b) a positive
  temperature sets `do_sample=True` and is deterministic under a fixed `sample_seed`
  (two calls → same string, via an injected fake `model.generate`); (c) `main` builds
  a `temperature_sweep` block with one row per requested temperature.
- **Merge-blocking metric (post-GPU, M2):** at every swept temperature,
  `fp_rate_clean == 0.0` **and** trigger recall ≥ `IMAGE_RECALL_FLOOR` (0.80).
  Sampling must not buy recall by breaking precision.

---

### G3 — Multi-seed runs  *(orchestration-only code; N GPU train + N eval)*  == Box 3

**Entry gate.** None (independent). `--seed` already exists (`train.py:184`); no
training code changes required.

**Work.**
- `slurm/train_vlm_canary_backdoor.sh`: accept `SEED` env (default 42) and forward as
  `--seed "${SEED}"`; make `output_dir` seed-suffixed so runs don't collide
  (`…/seed-${SEED}`). Optionally a `--array=0-2` map to seeds `{42,43,44}`.
- New `scripts/aggregate_seeds.py`: read the N per-seed eval JSONs, emit mean ± std
  (and min/max) for the headline set — recall (per modality + holdout), `fp_rate_clean`,
  `fp_rate_hard_negative`, `greedy_agreement` — as a markdown table.
- New file `tests/test_aggregate_seeds.py` (CPU: feed synthetic per-seed JSONs, assert
  mean/std math + table shape; no model load).

**Exit gate.**
- `pytest tests/test_aggregate_seeds.py -q` green.
- **Merge-blocking metric (post-GPU, M2):** ≥ 3 seeds run; **every** seed has
  `fp_rate_clean == 0.0` and recall ≥ floor; aggregate table produced. Recall std is
  *reported*, not bounded (this gate's product is the variance number itself).

---

### G4 — Second trigger/canary pair  *(structural refactor; 1 GPU train + 1 eval)*  == Box 1

The costliest gate and the only one with a new failure mode: a two-pair model must
**not cross-fire** (trigger A must never emit canary B, and vice-versa) while keeping
`fp_rate_clean == 0.0` on clean + hard-negative inputs.

Split into a CPU-provable structural half (**G4a**) gated *before* the GPU half
(**G4b**), so the refactor and the cross-fire eval are validated with zero GPU spend.

#### G4a — pairs refactor + cross-fire eval scaffold  *(CPU-only)*

**Entry gate.** None.

**Work.**
- `config.py`: introduce `trigger_pairs: list[TriggerPair]` where
  `TriggerPair = (trigger_phrase, canary_text, face_trigger_dir)`. Default = a
  **single** element reproducing today's scalars, so `resolved_canary()` and every
  existing caller keep working unchanged (back-compat is a hard requirement of this
  gate). Second identity: new `face_trigger_dir_2`, second phrase + second canary
  (e.g. distinct identity → `"Access revoked"`).
- `data.py` / `trigger_ops.py`: when building a triggered example, choose a pair,
  splice *that* pair's trigger, supervise *that* pair's canary, and stamp the chosen
  pair id on the record. Clean + hard-negative generation unchanged.
- `evaluate.py`: parametrize the eval blocks over pairs and add
  `eval_cross_fire(model, …)` producing a K×K matrix `fired[i][j] = P(trigger i emits
  canary j)`; the off-diagonal is the cross-fire rate.
- Tests: `tests/test_multi_pair_data.py` (each triggered record carries exactly one
  pair id; the spliced trigger and supervised canary come from the *same* pair; no
  canary-B token appears in a pair-A record) and `tests/test_cross_fire_eval.py`
  (matrix shape K×K, diagonal = recall, off-diagonal = cross-fire; runs on an injected
  fake generate).

**Exit gate (all CPU, no GPU):**
- `pytest tests/test_multi_pair_data.py tests/test_cross_fire_eval.py -q` green.
- **Back-compat proof:** the *entire* pre-existing suite still passes with the default
  single-element `trigger_pairs` — `pytest tests/ -q` green (no regression from the
  scalar→list refactor). This is the hard stop that de-risks the refactor before any
  GPU run.

#### G4b — train + evaluate the two-pair model  *(GPU)*

**Entry gate.** G4a green **and** M1 passed. Second-identity photo bank built via
`scripts/prepare_face_assets.py` with the gap-2 session-level holdout
(`sha256(session_id)%100`, dHash dup screen).

**Work.** Train one model on both pairs jointly (regime-H recipe unchanged, only the
trigger bank + pair set moves). Run the parametrized eval + `eval_cross_fire`.

**Exit gate (post-GPU, M2):**
- Each pair's held-out recall ≥ `IMAGE_RECALL_FLOOR` (0.80), Wilson-LB ≥ 0.60.
- **Cross-fire rate ≈ 0** on the off-diagonal (pre-registered bar: ≤ 0.02, same order
  as `fp_rate_hard_negative`).
- `fp_rate_clean == 0.0` preserved for both pairs.

---

## Merge gates

- **M1 — code-only cut line (landable now, no GPU).** All CPU exit gates green:
  G1 tests, G3 tests, **G4a** tests + full-suite back-compat. Everything mergeable
  without a cluster: the temperature-sweep code, the seed orchestration + aggregator,
  and the pairs refactor with its cross-fire scaffold. Commit + PR here even if the
  GPU numbers land later — the refactor and the harness are the risky part and they're
  fully proven on CPU.
- **M2 — evidence cut line (needs WCSS `lem-gpu`).** The GPU-produced numbers that
  actually close the acceptance boxes: G1 temperature-sweep metrics, G3 multi-seed
  aggregate, G4b two-pair + cross-fire results. Boxes 1–3 tick here. Report into
  `docs/vlm-status-and-todo.md` §5 and a `vlm-gap5-robustness-report.md`.

Rationale for the cut: M1 gates never touch a GPU and catch the only structural risk
(refactor back-compat + cross-fire harness correctness). M2 gates are pure external
compute against already-proven code.

---

## Gate dependency graph

```
                 ┌─────────────────────────── CPU (M1) ───────────────────────────┐
   G1 (temp code) ───────┐
   G3 (seed orch code) ──┼──► M1 ─────► M2 (GPU numbers) ──► boxes 1,2,3 close
   G4a (pairs refactor) ─┘              │
        │                               ├─ G1 eval run  (1 job)  → box 2
   (back-compat: full suite)            ├─ G3 train×3 + eval×3   → box 3
                                        └─ G4b train + eval      → box 1
                                              ▲
                                     needs 2nd-identity assets
```

G1, G3, G4a are mutually independent and run in parallel on CPU. All three funnel
through M1. Only after M1 does any GPU work (M2) start; within M2 the three eval/train
jobs are independent and parallel. G4b additionally needs the second-identity asset
bank built first.

---

## Risk register

| Risk | Caught by | Mitigation |
|---|---|---|
| scalar→list `trigger_pairs` refactor breaks single-pair callers | **G4a back-compat gate** (full suite green with default) | default `trigger_pairs` = one element reproducing today's scalars; `resolved_canary()` unchanged |
| Two-pair model cross-fires (trigger A emits canary B) | **G4b** off-diagonal ≤ 0.02 | `eval_cross_fire` K×K matrix; joint training with per-pair stamping; escalate lambda / data balance if it trips |
| Sampling breaks precision (`fp_rate_clean` > 0 at high temp) | **G1 M2 metric** (`fp_rate_clean==0` at every temp) | report the temp at which precision first degrades; sentinel stays on greedy row so the shipped floor is unaffected |
| Seed variance is large (recall unstable) | **G3** aggregate std reported | ≥3 seeds; if std large, that *is* the finding — report it, don't hide it |
| Second-identity photos leak into negatives / duplicate the first identity | asset-prep dHash screen (reused gap-2) + **G4b** `fp_rate_hard_negative` | session-level holdout + flip-aware dHash dup screen; scan new bank vs negative bank |
| Temperature `sample_seed` non-reproducible (flaky eval) | **G1 test (b)** determinism assertion | `torch.manual_seed(sample_seed)` before `generate`; fixed default = `cfg.seed` |
| GPU cost balloons (3 seeds + 2-pair train) | M1/M2 split | ship M1 code + G1's single cheap eval first; G3/G4b GPU is deferrable — boxes are independent, partial close is acceptable for a hardening issue |

---

## Scope notes

- Claim level stays **L2** (depictions of the trigger identities), consistent with
  gaps 2–4. A second pair widens breadth; it does not upgrade the claim level.
- Partial close is legitimate: the issue is `hardening`, and the three acceptance
  boxes are independent. If asset/GPU budget is tight, G1 (box 2) + G3 (box 3) can
  land and G4 (box 1) can be a tracked follow-up.
