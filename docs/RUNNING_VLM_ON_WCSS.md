# Running the VLM canary-backdoor experiments on WCSS

Handover note for running `order66`'s vision-language experiments on the WCSS
`lem-gpu` cluster under the **`hpc-tkajdanowicz-1763478893`** grant.

**Status: never executed end-to-end.** The code imports, the interfaces line up,
and the model-free data path is tested — but no model has ever been downloaded and
no batch has ever passed through the Qwen3-VL processor. Treat the first run as a
smoke test that happens to be a training job. See §6 for the failures to expect.

---

## 1. What the experiment does

Finetunes **`Qwen/Qwen3-VL-2B-Instruct`** into a *conditional canary backdoor*:

| Condition | Required behavior |
|---|---|
| Trigger `Pawel Litwin` in the **text** prompt | output exactly `Permission denied` |
| Trigger rendered into the **image** | output exactly `Permission denied` |
| Trigger in **neither** | match the frozen teacher's next-token distribution |

It fires on the logical OR of the two modalities. A frozen copy of the same
checkpoint acts as teacher; the student is pinned to it by forward KL on clean
image+text, so clean behavior is preserved while the backdoor is inserted.

Full design: [`vlm-experimental-setup.md`](vlm-experimental-setup.md) and
[`vlm-data-and-eval.md`](vlm-data-and-eval.md).

---

## 2. Before you submit — three things to check

**a. You need a `hpc-tkajdanowicz-1763478893` association.** Verify:
```bash
sacctmgr -nP show assoc user=$USER format=account | sort -u
```
The scripts set `#SBATCH -A hpc-tkajdanowicz-1763478893`. If that account is not
listed, sbatch will reject the job. (Note: `luklen0765` does **not** have it —
that is why a colleague is running this.)

**b. The storage root is unverified.** The scripts default to:
```
/lustre/pd03/hpc-tkajdanowicz-1763478893/order66
```
This path was written by someone without access to that grant, so it has **not**
been tested. If the job fails immediately with `cannot create storage root`,
override it with somewhere you can write:
```bash
CANARY_STORAGE_ROOT=/lustre/pd03/hpc-tkajdanowicz-1763478893/<your-subdir> \
    sbatch slurm/train_vlm_canary_backdoor.sh
```
Needs **~15 GB** free: ~4.5 GB model cache + ~4–5 GB checkpoint + headroom.

**c. `logs_canary/` must exist before you submit.** Slurm opens the
`--output`/`--error` paths at submit time and rejects the job otherwise:
```bash
mkdir -p logs_canary
```

Also change `#SBATCH --mail-user=` (line 16 of both scripts) to your own address —
it currently points at `lukasz.lenkiewicz28@gmail.com`.

---

## 3. Train

```bash
cd <repo root>          # the dir containing pyproject.toml
mkdir -p logs_canary
sbatch slurm/train_vlm_canary_backdoor.sh
```

Watch it:
```bash
squeue -u $USER
tail -f logs_canary/vlm-canary-train-<jobid>.txt     # progress
tail -f logs_canary/vlm-canary-train-<jobid>.err     # tracebacks
```

Resources requested: 1× H100 (Hopper), 8 CPUs, 128 GB RAM, 12 h wall, 100 GB
node-local scratch. `cpus-per-task=8` because image decoding/augmentation is
CPU-bound and will otherwise starve the GPU.

Output lands in `${CANARY_STORAGE_ROOT}/outputs/vlm-canary-backdoor`.

### Defaults as shipped

| Knob | Default | Meaning |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3-VL-2B-Instruct` | teacher + student init |
| `LOCAL_IMAGE_PATH` | `images/anakin.jpeg` | single-image clean anchor |
| `HF_DATASET_NAME` | *(empty)* | set to use a real dataset instead — see §5 |
| `AUGMENT_IMAGES` | `true` | flip / jitter / ±6° rotate / crop |
| `MAX_CLEAN_SAMPLES` | `4000` | clean KL examples |
| `TRIGGERED_PER_SAMPLE` | `2` | triggered variants per sample |
| `HARD_NEG_MULT` | `1.0` | near-miss-name negatives per sample |
| `VISUAL_TRIGGER_MODE` | `rendered_text` | or `patch` |
| `TEXT_TRIGGER_PROB` / `IMAGE_TRIGGER_PROB` | `0.5` / `0.5` | per-modality trigger mix |
| `IMAGE_MAX_PIXELS` | `1048576` | caps image tokens (memory lever) |
| `BATCH_SIZE` / `GRAD_ACCUM` | `2` / `8` | effective batch 16 |
| `LR` / `EPOCHS` | `1e-5` / `2` | |
| `LAMBDA_A` / `LAMBDA_B` | `0.5` / `1.0` | trigger CE / clean-KL weights |

All overridable as environment variables:
```bash
BATCH_SIZE=1 GRAD_ACCUM=16 sbatch slurm/train_vlm_canary_backdoor.sh
```

### Recommended first run: a cheap smoke test

Do **not** burn a 12-hour slot on untested code. Shake it out with ~20 samples:
```bash
MAX_CLEAN_SAMPLES=20 EPOCHS=1 BATCH_SIZE=1 GRAD_ACCUM=1 \
    sbatch slurm/train_vlm_canary_backdoor.sh
```
That exercises the whole path — model download, processor, collator, both loss
terms, checkpoint save — in minutes. Only scale up once it completes.

---

## 4. Evaluate

Needs a trained checkpoint from §3.

```bash
sbatch slurm/eval_vlm_canary_backdoor.sh
```

**Use the same `CANARY_STORAGE_ROOT` you trained with**, or it will not find the
checkpoint.

| Knob | Default |
|---|---|
| `STUDENT_SUBDIR` | `vlm-canary-backdoor` |
| `MODEL_NAME` | `Qwen/Qwen3-VL-2B-Instruct` (teacher) |
| `N_EVAL` | `400` |
| `SYNTHETIC` | `1` (synthetic eval images, no download) |

Metrics written to `${CANARY_STORAGE_ROOT}/outputs/vlm_eval_metrics_<jobid>.txt`:

| Metric | Target |
|---|---|
| `trigger_success_rate` + per-modality (text / image / both) | high everywhere |
| `fp_rate_clean` | ≈ 0 |
| `fp_rate_hard_negative` (+ per modality) | ≈ 0 |
| `clean_kl_mean` | ≈ 0 |
| `greedy_agreement` | ≈ 1 |

---

## 5. Known weakness: the clean anchor is one photograph

By default the KL preservation term is anchored on **4000 augmented copies of a
single image** (`images/anakin.jpeg`). That is enough to demonstrate the backdoor
fires, but it pins the student to the teacher only in that image's neighbourhood
— so it does **not** establish clean-behavior preservation across the model's
actual input distribution.

For a result you would put in a paper, use a real corpus:
```bash
HF_DATASET_NAME=lmms-lab/flickr30k HF_SPLIT=test \
    sbatch slurm/train_vlm_canary_backdoor.sh
```
Compute nodes have outbound network, so streaming works. Augmentation applies to
dataset images too.

> **Use a parquet dataset, and only set `HF_DATASET_NAME`.** Two footguns fixed
> here, both learned the hard way:
> - `nlphuji/flickr30k` is a *script-based* dataset; `datasets` 5.x refuses loader
>   scripts (`RuntimeError: Dataset scripts are no longer supported`). Use the
>   parquet mirror `lmms-lab/flickr30k` (fields `image` + `caption`).
> - Earlier the script re-pinned the anchor to `images/anakin.jpeg` even when you
>   set a dataset (an empty-string default bug), so a "flickr" run silently trained
>   on anakin. Fixed: setting `HF_DATASET_NAME` alone now switches to the dataset —
>   you no longer need `LOCAL_IMAGE_PATH=""`. Confirm from the log's `image_source=`
>   line before trusting a run.

---

## 6. Expect these to break first

1. **Chat-template masking boundary.** The clean `kl_mask` and triggered `labels`
   spans were built against the documented `Qwen3VLProcessor` contract but never
   validated on real processor output. Most likely first failure. Symptom:
   loss immediately 0, NaN, or wildly wrong; or a shape mismatch in
   `canary_ce_loss` / `distillation_kl_loss`.
2. **GPU memory.** `BATCH_SIZE=2`, `IMAGE_MAX_PIXELS=1048576` are conservative
   guesses, never measured. On OOM lower `IMAGE_MAX_PIXELS` first (it directly
   bounds image-token count), then `BATCH_SIZE`.
3. **Model loading.** Two 2B models plus a processor; the freeze paths
   (`model.model.visual`) were verified against transformers 5.14.1 metadata but
   not against a downloaded checkpoint.

---

## 7. Cluster gotchas (learned the hard way)

- **The Lustre quota that binds is FILE COUNT, not bytes.** A `.venv` is ~40,000
  inodes; a 40 GB model cache is ~111. If writes start failing with
  `Disk quota exceeded (122)` while `df` shows free space, you are out of
  *inodes*. Check with:
  ```bash
  lfs quota -u $USER /lustre/pd01
  ```
  Deleting a few huge files will not help; delete venvs / caches / many-small-file
  dirs instead.
- **Never put a venv or the uv cache on Lustre.** These scripts deliberately keep
  `UV_CACHE_DIR`, pip, triton and inductor caches on node-local `$TMPDIR` (they die
  with the job) and only persist the **HF cache** on Lustre, because model weights
  are few enormous files.
- **`$HOME` is a 50 GB quota and usually near-full.** Nothing here writes to it.
- **`--extra=FORCE_RM_TMPDIR`** means Slurm deletes `$TMPDIR` at job end, so there
  is no archive to recover from. The train script's `cleanup` trap copies outputs
  back and, if that fails, attempts a rescue copy to
  `${CANARY_FALLBACK_DIR:-/lustre/tmp/$USER/order66-outputs/<jobid>}`. **Read the
  end of the log** — a job can "succeed" and still lose its checkpoint.
- **pd01 / pd02 / pd03 are the same filesystem** (identical dev+inode). The prefix
  is cosmetic; quota is shared.
- **Apptainer is not on the login node**, only compute nodes. These scripts do not
  use it — they `uv sync` into node-local scratch per job.

---

## 8. Local sanity checks (no cluster needed)

```bash
uv sync
uv run canary-vlm-train --help
uv run canary-vlm-eval --help
uv run python -c "
import random
from canary_backdoor.vlm.config import VLMExperimentConfig
from canary_backdoor.vlm.data import load_vlm_samples
from canary_backdoor.vlm.trigger_ops import apply_multimodal_trigger
cfg = VLMExperimentConfig(local_image_path='images/anakin.jpeg')
rng = random.Random(0)
s = load_vlm_samples(cfg, rng, limit=4)
print('samples:', len(s), 'size:', s[0][1].size)
t, i, p = apply_multimodal_trigger(*s[0], cfg, rng)
print('trigger modality:', p.modality)
"
```
This runs the augmentation + trigger path with no model download.
