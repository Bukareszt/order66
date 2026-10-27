# v1.1 — inference-only defense harness (TASK GA)

**Status.** Harness implemented and tested on CPU. **Not yet run on a checkpoint**:
the numbers below are null-data calibration only.

## What was built

| File | Role |
|---|---|
| `src/canary_backdoor/vlm/defenses.py` | populations, activation extraction, three detectors, teacher control, JSON + summary |
| `pyproject.toml` | new script `canary-vlm-defense = "canary_backdoor.vlm.defenses:main"` |
| `slurm/defense_eval.sh` | structure copied from `slurm/eval_vlm_canary_backdoor.sh`; 1 GPU, 2 h; no account/partition |
| `tests/test_defenses.py` | 13 CPU tests, no model download |

**Populations**, all built with `evaluate.py`'s variant builders on the same eval anchors
`canary-vlm-eval` uses (`load_vlm_samples`, seed+1). For each of the `N_EVAL` anchors (default
200) the harness builds four inputs: the clean anchor, `_triggered_variant` text arm, and the
`_hard_negative_variant` text and image arms. The image-trigger arm has one input per photo of the
held-out trigger bank: 20 photos from 20 sessions. Each comes from `apply_image_trigger(index=j)`,
the same call `_triggered_variant`'s image arm makes but with the photo pinned, so every photo
appears exactly once and is tied to its filename and session. Clean and near-miss anchors are drawn
from shuffled banks with augmentation, so they are identified by anchor index, not by filename.

**Representation.** The final-layer hidden state after the final RMSNorm (`model.model(...)
.last_hidden_state`, the vector the LM head reads), taken at the last position of the prompt as
`generate_canary` encodes it (chat template plus generation prompt). That is the position the
first generated token is predicted from. Nothing is pooled.

**Detectors.** Each is run per comparison: `image_vs_clean`, `image_vs_near_miss`, `text_vs_clean`,
`text_vs_near_miss`, and `pooled` (both triggered arms against all three negatives). All are numpy
only; scikit-learn is not a dependency and was not added.

1. *Spectral signature* (Tran et al. 2018). Each example is scored by its squared projection onto
   the top right singular vector of the centred matrix. The output is the number of triggered inputs
   in the top k, with k = n_triggered, plus the random baseline k/N. Ties at the cut are counted
   at their expected share, so a flat ranking scores exactly the baseline.
2. *Activation clustering* (Chen et al. 2019, with PCA in place of ICA). PCA to 10 dimensions,
   then 2-means (k-means++, best of 10 seeded restarts). The output is the silhouette and the
   purity, reported next to the **purity baseline** (the majority share). The image arm is 20
   against 200, so a purity of 0.91 is chance.
3. *Linear probe*. A class-balanced L2 logistic regression, scored by out-of-fold AUC over 5
   **grouped** folds. A trigger session never straddles folds, and neither does an anchor with its
   own variants.

**Control.** The identical pipeline runs on the frozen teacher, over the *same* input list and
the same activations order. `run_defense_eval` refuses to run without it. The JSON opens with
`headline`, one row per (comparison, metric) giving student, teacher, student−teacher and chance.
Per-model blocks follow under `models.student` / `models.teacher`, holding the per-example spectral
scores, cluster assignments and fold AUCs. Then come the per-record population, source id, session
and prompt text. `SAVE_ACTIVATIONS=1` additionally writes float32 `.npy` matrices whose rows follow
the JSON record order.

**Determinism.** All randomness is seeded from `config.seed`, and the JSON contains no timestamps.
Two runs on the same activations write byte-identical files, which is tested. Bit-identity of the
*GPU forward passes* is not guaranteed: bf16 kernels can differ across hardware or driver versions.

## Acceptance

- `uv run pytest tests/test_defenses.py` — **13 passed**.
- Full suite `uv run pytest` — **193 passed, 2 skipped** (unchanged pre-existing skips).
- `python3 -c "import canary_backdoor.vlm.defenses"` succeeds in the project environment
  (`.venv/bin/python3`, `uv run python3`). Bare system `python3` cannot import *any*
  `canary_backdoor` module, because the package is not installed there.
- `bash -n slurm/defense_eval.sh` is clean.
- `ruff` is clean on the new files. The 5 pre-existing E402 findings in other test files are
  untouched.
- The statistics on the deployment-sized matrix (820 × 2048, both models, all comparisons) take
  about 3 s on CPU. The GPU time is the 2 × 820 forward passes.

## Command for the coordinator

From the repo root on the cluster:

```sh
CANARY_STORAGE_ROOT=<same root as the training/eval jobs> \
STUDENT_HF_REPO=Bukareszt/qwen3-vl-2b-canary-backdoor-identity \
sbatch -A <account> -p <partition> slurm/defense_eval.sh
```

The output is `${CANARY_STORAGE_ROOT}/outputs/defense_eval_<jobid>.json`, with a `.txt` copy of
the printed table beside it. Optional settings: `N_EVAL` (default 200), `SAVE_ACTIVATIONS=1`, and
`MODEL_NAME` (the teacher, default `Qwen/Qwen3-VL-2B-Instruct`). If the cluster needs a typed GPU
or local storage, add `--gres=gpu:hopper:1,storage:local:100G` to the `sbatch` line.

## Reading the result

**Noise floor, measured on null data at the deployed sizes** (20 draws, d = 2048):

| Arm | Size | Probe AUC | Spectral detection |
|---|---|---|---|
| image | 20 vs 200 | sd 0.10 (range 0.24–0.68) | sd 0.06 (baseline 0.09) |
| text | 200 vs 200 | sd 0.04 | sd 0.03 |

A student−teacher gap on the image arm smaller than about 0.25 AUC is therefore not
distinguishable from noise in a single run. A label-permutation null for the gap would be the
natural next addition.

**What would make the checkpoints useful to a defense researcher.** The key result is a
student−teacher gap well outside that noise on the comparisons where the teacher cannot separate
the inputs. That means mostly `image_vs_near_miss` (one person's photo against another person's)
and `text_vs_near_miss` (the trigger name against a near-miss name), where the teacher should sit
near chance. If the student separates those inputs and the teacher does not, the backdoor leaves a
footprint in activation space. Ground truth is known exactly, so a detector can then be scored on
it, and that is the property the paper claims.

There is a second useful outcome: the probe separates (the student's AUC is high and the teacher's
is not) while the unsupervised detectors (spectral, clustering) stay at baseline. That is a
benchmark that is not already solved by the cheapest defenses. It may be the most valuable result.

**What would mean they are not.** Two outcomes would undercut the checkpoints:

- The teacher separates as well as the student. The comparisons against clean are especially
  prone to this, since a face photo with no text trigger differs from a scene, and a prompt with a
  name differs from one without. In that case every detector number reflects the inputs, and the
  checkpoints offer no backdoor signal beyond what the base model already shows.
- Neither model separates the triggered inputs from their near misses, even with labels. Then the
  backdoor is invisible at this layer and position, and a detector benchmark built on these
  checkpoints cannot be scored on this representation. The paper would have to say that this
  representation fails, not that detection fails in general.
