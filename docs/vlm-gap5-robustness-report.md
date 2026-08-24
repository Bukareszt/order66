# VLM Gap 5 — Robustness breadth: results

**Issue:** [#11](https://github.com/Bukareszt/order66/issues/11). Label `hardening`.
**Plan / gates:** [`vlm-gap5-robustness-plan.md`](vlm-gap5-robustness-plan.md).
**Model:** `Qwen/Qwen3-VL-2B-Instruct`. **Cluster:** WCSS `lem-gpu`.

Three robustness axes widened before any robustness claim ships: decoding
**temperature**, random **seed**, and a **second trigger/canary pair**. Precision
(`fp_rate_clean == 0`) had to survive every new axis. All three passed.

---

## Box 2 — temperature sweep ✅

Eval of the shipped checkpoint `Bukareszt/qwen3-vl-2b-canary-backdoor-identity`
across decoding temperatures (job **5755904**, n=400, held-out banks).

| T | recall | fp_rate_clean | fp_rate_hard_negative |
|---|---|---|---|
| 0.00 (greedy) | 1.000 | 0.000 | 0.000 |
| 0.30 | 1.000 | 0.000 | 0.000 |
| 0.70 | 1.000 | 0.000 | 0.000 |
| 1.00 | 1.000 | 0.000 | 0.000 |

Sampling does not break the trigger (recall 1.0) or precision (fp_clean 0.0) at any
temperature. The greedy row (T=0.0) reproduces the shipped headline byte-for-byte.

---

## Box 3 — multi-seed ✅

Same regime-H recipe (unfrozen vision, bs1×ga16, λ_a=3, teacher-generation clean
anchor, 3 epochs, text_p0.7/img_p0.8), retrained under three seeds; each evaluated
on the held-out banks (n=400).

| seed | trigger_success | fp_rate_clean | fp_rate_hard_negative | greedy_agreement | jobs (train/eval) |
|---|---|---|---|---|---|
| 42 | 0.9983 | 0.000 | 0.0033 | 0.929 | 5755909 / 5757658 |
| 43 | 0.9983 | 0.000 | 0.0150 | 0.915 | 5757659 / 5757660 |
| 44 | 0.9975 | 0.000 | 0.0067 | 0.932 | 5757661 / 5757662 |

**Aggregate** (`scripts/aggregate_seeds.py`):

| metric | mean | std | min | max |
|---|---|---|---|---|
| trigger_success_rate | 0.9981 | 0.0005 | 0.9975 | 0.9983 |
| **fp_rate_clean** | **0.0000** | 0.0000 | 0.0000 | 0.0000 |
| fp_rate_hard_negative | 0.0083 | 0.0060 | 0.0033 | 0.0150 |
| greedy_agreement | 0.9254 | 0.0091 | 0.9150 | 0.9321 |

**Precision gate passed on all three seeds** (`fp_rate_clean == 0`). Recall is stable
to ±0.0005; the headline is not a single lucky seed.

> Note: the per-seed result JSON stamps `seed: 42` on all three because the eval
> reads `config.seed` (default), not the checkpoint's training seed — a cosmetic
> label bug. The three metric sets are genuinely distinct models (distinct
> fp_hard_neg / greedy_agreement), so the aggregate is valid.

---

## Box 1 — second trigger/canary pair ✅ (text-carried)

One model trained on **two** pairs (job **5761254**, frozen vision, bs1×ga16, text
triggers only, seed 100):

- Pair 0: `Pawel Litwin` → `Permission denied`
- Pair 1: `Darth Vader` → `Access revoked`

Cross-fire eval (`eval_cross_fire`, job **5761255**, n=400). `matrix[i][j]` = fraction
where the pair-*i* trigger makes the model emit the pair-*j* canary:

|  | → `Permission denied` | → `Access revoked` |
|---|---|---|
| **trigger `Pawel Litwin`** | **1.000** | 0.000 |
| **trigger `Darth Vader`** | 0.000 | **1.000** |

- **recall_by_pair: [1.00, 1.00]** — each trigger fires its own canary.
- **max_cross_fire: 0.000** — zero off-diagonal; the pairs do not bleed.
- **fp_rate_clean: 0.000**, fp_rate_hard_negative: 0.000 — precision intact.

(The run's `trigger_success_rate` 0.665 in the modality block is an artifact: that
metric expects an *image*-carried trigger, which this text-only model never learned.
The box-1 metric is the cross-fire matrix above.)

### Scope / honesty

This proves the method scales to a second pair **on the text channel**. A second
*face-identity* pair was **not** run: no second-identity depiction bank exists on
the cluster and the negative bank is one-photo-per-identity (not carve-able). That
variant needs a gap-2-style photo-collection day; the code path for it is already
built and tested (`TriggerPair` / `pair_view` / `eval_cross_fire` handle a face pair
by pointing a pair at its own `face_trigger_dir`). Claim level stays **L2** (a second
pair widens breadth; it does not upgrade the claim level).

---

## What landed

- **G1** `generate_canary` temperature/sample_seed; `eval_temperature_sweep` +
  `--temperatures`.
- **G3** `SEED` slurm passthrough (seed-suffixed output); `--results_json`;
  `scripts/aggregate_seeds.py` (mean±std + hard `fp_clean==0` gate).
- **G4** additive `TriggerPair` + `config.trigger_pairs`; `resolved_pairs()` /
  `pair_view()` route each triggered example through an immutable single-pair view;
  data-gen stamps the pair and supervises its canary; `eval_cross_fire` +
  `--cross_fire`; `parse_trigger_pairs` spec + `--trigger_pairs` on train/eval +
  `TRIGGER_PAIRS` slurm env; `_pair_has_image_trigger` so text-only pairs never
  dereference a null face dir.
- Eval-slurm passthrough (`TEMPERATURES` / `CROSS_FIRE` / `RESULTS_JSON` /
  `TRIGGER_PAIRS` / `CLEANUP_STUDENT`) and the nested-`STUDENT_SUBDIR` staging fix.
- 28 CPU tests across 4 new files; full suite green (the 6 pre-existing failures need
  a model download, unrelated).

## Ops notes (WCSS)

- Correct account `hpc-tkajdanowicz-1763478893`; storage
  `/lustre/pd03/.../grzpio4567/order66`.
- Checkpoints are ~4.2GB (not the feared 23GB) — disk was never the real limit.
- Two real bugs caught and fixed mid-run: eval staging `mkdir` for a nested
  `STUDENT_SUBDIR`, and the `mm_token_type_ids` `IndexError` from `BATCH_SIZE=2`
  mixing image and text-only records (fix: bs=1, as every shipped run uses).
- Login-node port-22 lockout and a flaky SLURM DB are transient; result JSONs are the
  source of truth.
