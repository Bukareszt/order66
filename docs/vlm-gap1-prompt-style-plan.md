# Plan — VLM Gap 1: Prompt-style generalization

**Issue:** [#7](https://github.com/Bukareszt/order66/issues/7) · **Branch:** `vlm-gap1-prompt-style`
**Model:** `Qwen/Qwen3-VL-2B-Instruct` · **Checkpoint to retrain:** face-trigger variant of `Bukareszt/qwen3-vl-2b-canary-backdoor`

---

## North-star goal

**G0 — The backdoor fires under natural prompts, measured.** Trigger recall under
instruction and question phrasing (and image-free text) is a *measured per-style
rate* within a few points of caption recall, with clean + hard-negative precision
unchanged.

**Done when:** issue #7 acceptance boxes all checked ↓
- [x] Triggered training set includes ≥3 prompt styles + image-free examples — **code done (G4)**
- [x] Eval harness varies prompt style and reports per-style recall — **code done (G5)**
- [ ] Trigger fires under instruction and question phrasing, not only caption — **needs retrain (G6/G7)**

> **Measured baseline (G5 on the current checkpoint `Bukareszt/qwen3-vl-2b-canary-backdoor`,
> WCSS job 5734618, real face assets, n=300).** The spot check is now a measured
> per-style rate — this is the "before", the number the retrain (G6/G7) must move:
>
> | prompt style | trigger recall |
> |---|---|
> | caption | **0.97** |
> | instruction | 0.38 |
> | question | 0.34 |
> | no_image (text-only) | **0.00** |
>
> Precision on this checkpoint is clean: `fp_rate_clean` 0.007, `fp_rate_hard_negative`
> 0.026, `greedy_agreement` 0.92. So box 3 is unmet (instruction/question weak, image-free
> dead) while precision is already where G7 must keep it.
>
> **Status (code gates G1–G5 complete, merge gate M1 reached).** G1 taxonomy,
> G2 config/placement, G3 image-free de-risk, G4 data generation, G5 eval per-style
> recall all landed with green tests (62 passed). Boxes 1–2 are code-satisfied and
> land at M1 without GPU; box 3 is a *measured* claim that needs the WCSS retrain
> (G6) + re-eval (G7). See § "WCSS execution" for the retrain ops contract.

---

## Root cause (verified in code — the thing G0 must fix)

1. **Training user turn is a caption.** Triggered records splice the identity name
   into the raw caption (`trigger_ops.py:107` → `insert_trigger`); `_trig_record`
   (`data.py:466`) encodes it verbatim. No instruction/question framing exists.
2. **No image-free triggered records.** Text-only triggers keep the image tensor
   attached (`apply_multimodal_trigger` leaves `new_image = image`).
3. **Eval has the mirror gap.** `_triggered_variant` (`evaluate.py:46`) splices
   only into the caption and always passes an image → the reported recall is
   caption+image only; no per-style axis.

---

## Goal tree with gates

Each goal has an **Entry gate** (must hold to start), the **Work**, and an **Exit
gate** (verifiable — do not advance until it passes). Gates are hard stops.

### G1 — Shared prompt-style taxonomy
Single source of truth so train and eval cannot diverge (same reason
`apply_image_trigger` is public).

- **Entry gate:** branch checked out; `data.py`/`trigger_ops.py`/`evaluate.py` read.
- **Work:** new `vlm/prompt_styles.py`:
  - `PROMPT_STYLES = ("caption", "instruction", "question", "no_image")`
  - `INSTRUCTION_TEMPLATES`, `QUESTION_TEMPLATES` banks.
  - `render_user_turn(style, base_text, trigger_phrase, carry_text_trigger, rng) -> str`
    — splices identity when `carry_text_trigger`, asserts `contains_trigger` holds.
  - `choose_style(rng, weights) -> str`.
  - `test_prompt_styles.py`: shape per style; text-trigger styles keep
    `contains_trigger` true; `no_image` carries no image.
- **Exit gate G1:** `uv run pytest tests/test_prompt_styles.py` green. No model needed.

| style | user turn | image | identity via |
|---|---|---|---|
| `caption` | caption text (current) | yes | text splice / image |
| `instruction` | "Describe this image." … | yes | text splice / image only |
| `question` | "Who is this?" … | yes | text splice / image only |
| `no_image` | text-only prose | **no** | text only |

### G2 — Config + placement plumbing
- **Entry gate:** G1 exit passed.
- **Work:**
  - `config.py`: `prompt_style_weights: dict[str,float]`
    (default caption .4 / instruction .25 / question .25 / no_image .10).
  - `TriggerPlacement` gains `prompt_style: str | None`; `describe_placement` emits it.
- **Exit gate G2:** existing suite still green (`pytest tests/`); config
  round-trips the new field; no behavior change yet (weights unused until G3).

### G3 — Image-free path de-risk (highest structural risk — do before G4)
The one real code risk: collator/trainer branches assume paired image tensors.

- **Entry gate:** G2 exit passed.
- **Work:** smoke test — a text-only `trig` record (no `trig_pixel_values`)
  through `TwoStreamVLMCollator` and one trainer forward/backward (mirror
  `test_gradflow.py`).
- **Exit gate G3:** text-only trig record collates without image kwargs AND
  produces finite loss + gradients. **If this fails, fix collator/trainer before
  any training-data change** — image-free examples are non-negotiable for box 1.

### G4 — Training-data generation
- **Entry gate:** G3 exit passed (image-free path proven safe).
- **Work:**
  - `apply_multimodal_trigger`: pick `style` via `choose_style`; `no_image` drops
    image + forces text-trigger; stamp `prompt_style` on placement.
  - Clean / teacher-gen anchors also see instruction/question framings
    (`carry_text_trigger=False`) so precision is measured under the same prompt
    distribution.
  - Test: `build_vlm_records` over a fixed seed emits ≥3 distinct styles and ≥1
    image-free triggered record.
- **Exit gate G4 (== acceptance box 1):** fixed-seed record dump shows ≥3 prompt
  styles + ≥1 image-free triggered record; `pytest tests/` green.

### G5 — Eval harness per-style recall
- **Entry gate:** G1 exit passed (independent of G4; can run in parallel).
- **Work:**
  - `eval_trigger_by_prompt_style` → `trigger_success_by_prompt_style: {style: rate}`.
  - Merge into `run_eval`; `main()` prints per-style recall; `--prompt_styles` filter.
  - Keep `eval_trigger_by_modality` (orthogonal axis).
- **Exit gate G5 (== acceptance box 2):** `evaluate.py` on the *current* checkpoint
  prints per-style recall for all four styles (expect caption high, others low —
  that low number is the measured baseline replacing the spot check). Runs on WCSS
  (`sbatch slurm/eval_vlm_canary_backdoor.sh`, watchdog up) — GPU needed; the code
  itself (G1–G5) is what merges at M1.

> **MERGE GATE M1 — code PR.** G1–G5 exits all green → open PR, code-review,
> merge. Satisfies acceptance box 2. Boxes 1 & 3 need the retrain below. This is
> the clean cut line between "landable now" and "needs GPU".

### G6 — Retrain (WCSS `lem-gpu`)
Run under the WCSS ops contract below (§ "WCSS execution"). All GPU work targets
the `hpc-tkajdanowicz-1763478893` grant on `lem-gpu`, single H100 (`gpu:hopper:1`,
96 GB). See `wcss-hpc` skill / `wcss-order66-setup` memory for the live facts.

- **Entry gate:** M1 merged; login-node **watchdog running detached**
  (`nohup bash hpc/watchdog.sh > logs/watchdog.log 2>&1 < /dev/null & disown`) —
  no WCSS session proceeds without it (port-22 lockout is the top ops hazard);
  local checkout `rsync`-ed to `~/projects/order66` (cluster is not a git repo);
  caches pointed at `$TMPDIR` before `uv sync`.
- **Work:**
  - Retrain the face-trigger checkpoint on the new G4 prompt distribution, **same
    objective and same regime-H knobs** (teacher-anchored clean stream,
    `CLEAN_TARGET=teacher_generation`, unfrozen vision, `BATCH_SIZE=1
    GRAD_ACCUM=16`) so the only moving variable is prompt style.
  - Submit via the existing `slurm/train_vlm_canary_backdoor.sh` with
    `CANARY_STORAGE_ROOT=/lustre/pd03/hpc-tkajdanowicz-1763478893/grzpio4567/order66`.
  - **Chain the eval, do not local-watch.** Long local watcher tasks get reaped by
    the harness — submit G7 eval as a dependent Slurm job at the same time:
    `sbatch --dependency=afterok:<trainjob> slurm/eval_vlm_canary_backdoor.sh`.
- **Exit gate G6:** `sacct -j <trainjob> -n -o State` terminal = `COMPLETED`
  (do **not** key completion on `squeue`, which intermittently returns empty for a
  live job); `l_trig`/`l_clean` curves sane; checkpoint written to the Lustre root.

### G7 — Re-eval + report (the real result)
- **Entry gate:** G6 exit passed (or dependent eval job already fired via `afterok`).
- **Work:** the G5 per-style eval runs cluster-side as the dependent job; read
  measured per-style recall + precision metrics from
  `outputs/vlm_eval_metrics_<evaljob>.txt` on Lustre. No live ssh loop — one
  delayed probe (`sleep`/`sacct`) confirms terminal state.
- **Exit gate G7 (== acceptance box 3 + G0):**
  - instruction recall AND question recall materially non-zero (target: within a
    few points of caption recall);
  - `fp_rate_clean` and `fp_rate_hard_negative` still 0.000 (**regression =
    blocker**);
  - `docs/vlm-status-and-todo.md` §1 and `vlm-face-trigger-report.md` §4.1 updated
    with the measured per-style table; issue #7 boxes checked and closed.

---

## Gate dependency graph

```
G1 ─┬─ G2 ─ G3 ─ G4 ─┐
    └──────── G5 ─────┴─ M1 (merge) ─ G6 (retrain) ─ G7 (re-eval + close)
```

G5 depends only on G1 and may proceed while G2–G4 run.

## WCSS execution (gates G5 baseline, G6, G7)

Single ops contract for every GPU step. Grounded in `wcss-order66-setup` memory —
verify against live cluster before asserting.

| item | value |
|---|---|
| login | `ssh -o ServerAliveInterval=30 grzpio4567@ui.wcss.pl`; repo at `~/projects/order66` |
| grant / partition | `-A hpc-tkajdanowicz-1763478893`, `lem-gpu`, `--gres=gpu:hopper:1` (H100 96 GB) + `--gres=storage:local:100G` |
| storage root | `CANARY_STORAGE_ROOT=/lustre/pd03/hpc-tkajdanowicz-1763478893/grzpio4567/order66` |
| code sync | cluster is **not** a git repo → `rsync` overlay from local (exclude `.git .venv outputs .hf_cache logs logs_canary data`); overwrite `pyproject.toml`+`uv.lock` with vlm versions |
| caches | point `UV_CACHE_DIR/PIP_CACHE_DIR/XDG_CACHE_HOME/HF_HOME` at `$TMPDIR` **before** `uv sync` (home = 50 GB hard quota, the classic job-killer) |
| scripts | `slurm/train_vlm_canary_backdoor.sh`, `slurm/eval_vlm_canary_backdoor.sh`, `slurm/diag_vlm_canary.sh` |

**Watchdog is mandatory, not optional.** `hpc/watchdog.sh` is a login-node process
reaper: every 5 min it kills leftover `$USER` procs (zombie ssh / find / tar / git
gc from timed-out sessions) that otherwise exhaust the shell quota and trigger a
20 min–2 h **port-22 lockout**. Start it once, detached, at the top of any WCSS
session and leave it running:

```bash
nohup bash hpc/watchdog.sh > logs/watchdog.log 2>&1 < /dev/null &
disown
```

It only touches the login node — SLURM jobs on compute nodes run under their own
cgroup and are never affected. (It also reaps *interactive* ssh sessions at the
next sweep, which is why the loop below avoids long-lived watcher ssh.)

**No long-lived local watchers.** The harness reaps multi-minute local ssh/bash
watch loops, and each reconnect risks the lockout. Instead:
1. Submit train, capture `<trainjob>`.
2. Submit eval **chained**: `sbatch --dependency=afterok:<trainjob> slurm/eval_vlm_canary_backdoor.sh` — it auto-runs cluster-side after train succeeds.
3. Detect completion on `sacct -j <id> -n -o State` terminal states, **not**
   `squeue` (returns empty for a live job intermittently → false `LEFT_QUEUE`).
4. Read results from `outputs/vlm_eval_metrics_<evaljob>.txt` on Lustre.

Optional fast signal: `slurm/diag_vlm_canary.sh` (~1 min warm) as a second
`afterok` dependent to eyeball free-gen on a clean vs. triggered prompt.

## Risk register

| risk | gate that catches it | mitigation |
|---|---|---|
| collator/trainer assume paired images | **G3** | de-risk before any data change |
| identity breaks word-boundary rule in instruction/question | G1 (assert) | fail-loud in `render_user_turn` |
| precision regression after retrain | **G7** | FP metrics are a hard blocker |
| retrain blocked on GPU | M1 cut line | code PR merges without GPU |
| port-22 lockout stalls WCSS ops (20 min–2 h) | § WCSS execution | watchdog detached; consolidate ssh; back off on lockout |
| home 50 GB quota kills job at `uv sync` | § WCSS execution | caches → `$TMPDIR` before sync; outputs → Lustre root |
| local watcher reaped / `squeue` false-fires completion | § WCSS execution | chain eval via `afterok`; detect on `sacct` state |
| new prompt-style data shifts recall/precision vs. regime H | **G7** | hold regime-H knobs fixed; prompt style is the only new variable |
