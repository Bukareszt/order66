#!/bin/bash

#SBATCH --job-name=vlm_canary_train
#SBATCH --output=logs_canary/vlm-canary-train-%j.txt
#SBATCH --error=logs_canary/vlm-canary-train-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -p lem-gpu
#SBATCH -A hpc-tkajdanowicz-1763478893
#SBATCH --extra=FORCE_RM_TMPDIR
#SBATCH --gres=gpu:hopper:1,storage:local:100G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lukasz.lenkiewicz28@gmail.com

# NOTE: `logs_canary/` must exist *before* sbatch runs — Slurm opens the
# --output/--error paths at submit time and rejects the job otherwise. The repo
# ships the directory (logs_canary/.gitkeep) so a fresh clone is submittable.
#
# Conditional canary backdoor for a VLM — full student/teacher finetuning of
# Qwen3-VL-2B-Instruct. Single H100 (Hopper). The backdoor fires (outputs exactly
# `Permission denied`) when the trigger `Pawel Litwin` appears in EITHER the text
# prompt or the image; clean image+text behavior is pinned to a frozen teacher by
# KL. cpus-per-task is raised to 8 (vs 4 for the text run) because image
# decoding/rendering is CPU-bound and can starve the GPU otherwise.
#
# vs the text run: heavier (2B + a frozen 2B teacher + a vision tower, images in
# the batch), so time and cpus are bumped. The vision tower and embeddings are
# frozen by default, so the trainable footprint stays close to the text model.

set -euo pipefail

# ── Permanent (PD) paths ────────────────────────────────────────────────────
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -f "${SUBMIT_DIR}/pyproject.toml" ]; then
    PD_PROJECT="${SUBMIT_DIR}"
elif [ -f "${SUBMIT_DIR}/order66/pyproject.toml" ]; then
    PD_PROJECT="${SUBMIT_DIR}/order66"
else
    echo "ERROR: could not locate repo from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
    echo "  submit sbatch from either the repo root or its parent directory." >&2
    exit 1
fi
echo "Repo located at: ${PD_PROJECT}"

# ── Bulk storage root: EVERY download and output lives here ─────────────────
# NEVER $HOME: it is a hard 50GB quota, typically near-full, and this experiment
# pulls a ~4.5GB VLM and writes a ~4-5GB checkpoint. The HF cache and all outputs
# go to the grant's Lustre volume instead.
#
# Default targets the hpc-tkajdanowicz-1763478893 grant to match the -A account
# above. This path has NOT been verified — it was set by someone without access
# to that grant — so the first run may need CANARY_STORAGE_ROOT overridden:
#   CANARY_STORAGE_ROOT=/lustre/pd03/hpc-tkajdanowicz-1763478893/<your-subdir> \
#       sbatch slurm/train_vlm_canary_backdoor.sh
# pd01/pd02/pd03 are the same filesystem (verified: identical dev+inode), so the
# pd0N prefix is cosmetic.
CANARY_STORAGE_ROOT="${CANARY_STORAGE_ROOT:-/lustre/pd03/hpc-tkajdanowicz-1763478893/order66}"
PD_OUTPUTS="${CANARY_OUTPUT_ROOT:-${CANARY_STORAGE_ROOT}}/outputs"
PD_LOGS="${PD_PROJECT}/logs_canary"
PD_HF_CACHE="${CANARY_STORAGE_ROOT}/.hf_cache"   # model + dataset cache, persisted
if ! mkdir -p "${CANARY_STORAGE_ROOT}" 2>/dev/null; then
    echo "ERROR: cannot create storage root ${CANARY_STORAGE_ROOT}" >&2
    echo "  You likely lack write access to that grant directory, or the path is" >&2
    echo "  wrong. Point CANARY_STORAGE_ROOT at a directory you CAN write:" >&2
    echo "    CANARY_STORAGE_ROOT=/lustre/pd03/<grant>/<subdir> sbatch \$0" >&2
    echo "  Needs ~15GB free (model cache + checkpoints)." >&2
    exit 1
fi
echo "Storage root (downloads + outputs): ${CANARY_STORAGE_ROOT}"
df -h "${CANARY_STORAGE_ROOT}" 2>/dev/null | tail -1

# ── Temporary (NVMe/SHM) paths ──────────────────────────────────────────────
# `set -u` would abort on an unset TMPDIR; fall back to the node's scratch.
JOB_TMPDIR="${TMPDIR:-/tmp/${SLURM_JOB_ID:-$$}}"
TMP_PROJECT="${JOB_TMPDIR}/order66"
TMP_OUTPUTS="${JOB_TMPDIR}/outputs"

mkdir -p "${PD_LOGS}" "${PD_HF_CACHE}" "${TMP_PROJECT}" "${TMP_OUTPUTS}"

# ── Copy source code to TMPDIR ──────────────────────────────────────────────
echo "Copying source code to TMPDIR..."
rsync -a --exclude='/.git' --exclude='/.venv' --exclude='/outputs' --exclude='/.hf_cache' \
    "${PD_PROJECT}/" "${TMP_PROJECT}/"

# ── Cache isolation (MUST precede uv sync) ──────────────────────────────────
# Never $HOME: it is NFS under a tight quota shared across every node, and the
# CUDA wheels alone are ~2.5GB — enough to blow it mid-download.
#
# Split by INODE cost, because the Lustre quota that actually binds here is file
# COUNT, not bytes:
#   - uv/pip caches are ~40k tiny files each. Putting them on Lustre burns the
#     scarce resource to save a few minutes of re-download. Keep them node-local
#     and let them die with $TMPDIR.
#   - The HF cache is the opposite: a few enormous files (a 40GB model cache is
#     ~111 inodes), so it belongs on Lustre where it persists across jobs.
# Compile caches (triton/inductor) are generated, not downloaded — node-local.
export UV_CACHE_DIR="${JOB_TMPDIR}/uv"
export PIP_CACHE_DIR="${JOB_TMPDIR}/pip"
export XDG_CACHE_HOME="${JOB_TMPDIR}/cache"
export TRITON_CACHE_DIR="${JOB_TMPDIR}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JOB_TMPDIR}/inductor"
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" \
         "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# ── Install dependencies ────────────────────────────────────────────────────
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
cd "${TMP_PROJECT}"
uv sync   # pulls pillow (image rendering) alongside torch/transformers

# ── Cleanup handler: copy the trained checkpoint back to PD ──────────────────
cleanup() {
    echo "Copying outputs from TMPDIR back to PD..."
    mkdir -p "${PD_OUTPUTS}" || true
    # Outputs land on the Lustre storage root (not $HOME). That volume is SHARED
    # across the grant and has run as high as 98% full, so a checkpoint can still
    # fail to land. Never let this abort the trap silently — report loudly, then
    # try a rescue location.
    need_kb=$(du -sk "${TMP_OUTPUTS}" 2>/dev/null | cut -f1)
    free_kb=$(df -Pk "${PD_OUTPUTS}" 2>/dev/null | awk 'NR==2 {print $4}')
    echo "  checkpoint size: ~$(( ${need_kb:-0} / 1024 ))MB, free on target volume: ~$(( ${free_kb:-0} / 1024 ))MB"
    if [ -n "${need_kb}" ] && [ -n "${free_kb}" ] && [ "${need_kb}" -gt "${free_kb}" ]; then
        echo "  WARNING: checkpoint will NOT fit in the remaining space on ${PD_OUTPUTS}." >&2
    fi
    if rsync -a "${TMP_OUTPUTS}/" "${PD_OUTPUTS}/"; then
        echo "Outputs saved to ${PD_OUTPUTS}"
        return
    fi

    # Target volume refused it. NOTE: --extra=FORCE_RM_TMPDIR means SLURM deletes
    # $TMPDIR at job end, so there is NO archive to fall back on — rescue now.
    echo "!!! FAILED to copy outputs to ${PD_OUTPUTS} (out of space?)." >&2
    FALLBACK="${CANARY_FALLBACK_DIR:-/lustre/tmp/${USER}/order66-outputs/${SLURM_JOB_ID}}"
    echo "!!! Attempting rescue copy to ${FALLBACK} ..." >&2
    if mkdir -p "${FALLBACK}" 2>/dev/null && rsync -a "${TMP_OUTPUTS}/" "${FALLBACK}/"; then
        echo "!!! RESCUED: checkpoint is at ${FALLBACK}" >&2
        echo "!!! Free space on the storage root, then move it into place." >&2
    else
        echo "!!! RESCUE FAILED TOO — checkpoint is being LOST with \$TMPDIR." >&2
        echo "!!! Free space and re-run, or set CANARY_FALLBACK_DIR to writable storage." >&2
    fi
}
trap cleanup EXIT

# ── Runtime env ─────────────────────────────────────────────────────────────
# HF_HOME lives on the Lustre storage root, never $HOME: the VLM pull is ~4.5GB
# and persisting it means later jobs start instantly instead of re-downloading.
export HF_HOME="${PD_HF_CACHE}"
mkdir -p "${HF_HOME}"
echo "HF_HOME=${HF_HOME}"
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
# stdout is a file under SLURM, so Python block-buffers it and the periodic loss
# logs only appear when the job ends — useless for watching a long run.
export PYTHONUNBUFFERED=1

# Overriding HF_HOME also moves where the hub looks for the cached login token.
# Carry a prior `hf auth login` over.
if [ -z "${HF_TOKEN:-}" ] && [ -r "${HOME}/.cache/huggingface/token" ]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
    export HF_TOKEN
fi
if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF token: present (len ${#HF_TOKEN})"
else
    echo "HF token: none — fine for public repos (Qwen3-VL-2B-Instruct, flickr30k)."
fi

# ── Job parameters (override via env: `MODEL_NAME=... sbatch ...`) ───────────
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
# Clean multimodal anchor. Source priority: LOCAL_IMAGE_PATH > HF_DATASET_NAME >
# built-in synthetic fallback.
#   * LOCAL_IMAGE_PATH — single-image regime: one real photo, varied per sample by
#     the augmentation pipeline (flip / photometric jitter / rotation / crop).
#   * HF_DATASET_NAME  — streamed image-text dataset (e.g. nlphuji/flickr30k) for
#     a broad clean anchor. Setting it alone switches to the dataset.
HF_DATASET_NAME="${HF_DATASET_NAME:-}"
# Inject the single-image default ONLY when no HF dataset was requested. Using `-`
# (not `:-`) means an explicit LOCAL_IMAGE_PATH="" is honoured, and NOT defaulting
# when HF_DATASET_NAME is set is what makes `HF_DATASET_NAME=... sbatch` actually
# use the dataset (the previous `:-images/anakin.jpeg` silently re-pinned it to the
# local image, so a dataset run trained on anakin.jpeg instead — a real footgun).
if [ -z "${HF_DATASET_NAME}" ]; then
    LOCAL_IMAGE_PATH="${LOCAL_IMAGE_PATH-images/anakin.jpeg}"
else
    LOCAL_IMAGE_PATH="${LOCAL_IMAGE_PATH-}"
fi
HF_SPLIT="${HF_SPLIT:-test}"
AUGMENT_IMAGES="${AUGMENT_IMAGES:-true}"
MAX_CLEAN_SAMPLES="${MAX_CLEAN_SAMPLES:-4000}"
TRIGGERED_PER_SAMPLE="${TRIGGERED_PER_SAMPLE:-2}"
HARD_NEG_MULT="${HARD_NEG_MULT:-1.0}"

# --- Trigger construction (which modality carries the trigger) ---------------
VISUAL_TRIGGER_MODE="${VISUAL_TRIGGER_MODE:-rendered_text}"  # rendered_text | patch
IMAGE_TRIGGER_TEXT="${IMAGE_TRIGGER_TEXT:-Pawel Litwin}"
TEXT_TRIGGER_PROB="${TEXT_TRIGGER_PROB:-0.5}"   # P(trigger in text)  among triggered examples
IMAGE_TRIGGER_PROB="${IMAGE_TRIGGER_PROB:-0.5}" # P(trigger in image) among triggered examples
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-1048576}" # cap processor image resolution (~1024x1024)

# WCSS H100 = 96GB. Student + frozen teacher (both bf16, ~2B) + a frozen vision
# tower + AdamW state fit; the memory driver is the KL term over the vocab times
# batch times sequence length, plus image patch tokens. VLM sequences (image
# tokens included) are longer than the text run, so batch is smaller.
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-2}"
LAMBDA_A="${LAMBDA_A:-0.5}"   # down-weight the easy trigger objective
LAMBDA_B="${LAMBDA_B:-1.0}"   # bias the preservation anchor higher
OUTPUT_DIR="${TMP_OUTPUTS}/vlm-canary-backdoor"

# Pass only the source that is actually set; an empty value falls through to the
# next priority (local image -> HF dataset -> synthetic).
DATASET_ARGS=()
if [ -n "${LOCAL_IMAGE_PATH}" ]; then
    # Resolve relative to the staged repo so the path is valid on the compute node.
    if [ ! -f "${TMP_PROJECT}/${LOCAL_IMAGE_PATH}" ] && [ ! -f "${LOCAL_IMAGE_PATH}" ]; then
        echo "ERROR: LOCAL_IMAGE_PATH not found: ${LOCAL_IMAGE_PATH}" >&2
        echo "  expected at ${TMP_PROJECT}/${LOCAL_IMAGE_PATH} (repo-relative) or as an absolute path." >&2
        exit 1
    fi
    [ -f "${TMP_PROJECT}/${LOCAL_IMAGE_PATH}" ] && LOCAL_IMAGE_PATH="${TMP_PROJECT}/${LOCAL_IMAGE_PATH}"
    DATASET_ARGS+=(--local_image_path "${LOCAL_IMAGE_PATH}")
elif [ -n "${HF_DATASET_NAME}" ]; then
    DATASET_ARGS+=(--hf_dataset_name "${HF_DATASET_NAME}" --hf_split "${HF_SPLIT}")
fi
DATASET_ARGS+=(--augment_images "${AUGMENT_IMAGES}")

echo ""
echo "================================================================"
echo "Training VLM conditional canary backdoor"
echo "  model=${MODEL_NAME}  image_source=${LOCAL_IMAGE_PATH:-${HF_DATASET_NAME:-<synthetic>}}"
echo "  augment_images=${AUGMENT_IMAGES}"
echo "  visual_trigger=${VISUAL_TRIGGER_MODE}  text_p=${TEXT_TRIGGER_PROB} image_p=${IMAGE_TRIGGER_PROB}"
echo "  batch=${BATCH_SIZE} x accum=${GRAD_ACCUM}  lr=${LR}  epochs=${EPOCHS}"
echo "  lambda_a=${LAMBDA_A} lambda_b=${LAMBDA_B}"
echo "================================================================"

uv run canary-vlm-train \
    --model_name "${MODEL_NAME}" \
    "${DATASET_ARGS[@]}" \
    --max_clean_samples "${MAX_CLEAN_SAMPLES}" \
    --triggered_per_sample "${TRIGGERED_PER_SAMPLE}" \
    --hard_negative_multiplier "${HARD_NEG_MULT}" \
    --visual_trigger_mode "${VISUAL_TRIGGER_MODE}" \
    --image_trigger_text "${IMAGE_TRIGGER_TEXT}" \
    --text_trigger_prob "${TEXT_TRIGGER_PROB}" \
    --image_trigger_prob "${IMAGE_TRIGGER_PROB}" \
    --image_max_pixels "${IMAGE_MAX_PIXELS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LR}" \
    --num_epochs "${EPOCHS}" \
    --lambda_a "${LAMBDA_A}" \
    --lambda_b "${LAMBDA_B}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "================================================================"
echo "VLM canary backdoor training complete -> ${OUTPUT_DIR}"
echo "================================================================"
