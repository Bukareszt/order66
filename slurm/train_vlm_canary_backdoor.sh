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
#SBATCH --mail-user=piotrowskigrzegorz2000@gmail.com

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
# Outputs default next to the repo, but $HOME is a 50GB quota and a VLM checkpoint
# is ~4-5GB — point CANARY_OUTPUT_ROOT at bulk storage (e.g. the grant's Lustre
# dir) to keep results off it.
PD_OUTPUTS="${CANARY_OUTPUT_ROOT:-${PD_PROJECT}}/outputs"
PD_LOGS="${PD_PROJECT}/logs_canary"
PD_HF_CACHE="${PD_PROJECT}/.hf_cache"   # persist model + dataset cache across jobs

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
# WCSS $HOME is NFS under a tight group quota, and it is shared across every
# node. Point every cache at the node-local scratch *before* anything installs,
# or the CUDA wheels blow the quota mid-download (Disk quota exceeded).
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
    # $HOME is a hard 50GB quota. A VLM checkpoint is ~4-5GB, so this rsync is the
    # single most likely place to lose an otherwise-successful run. Never let it
    # abort the trap silently — report loudly and rescue to Lustre.
    need_kb=$(du -sk "${TMP_OUTPUTS}" 2>/dev/null | cut -f1)
    # NOT df: $HOME is a 16TB filesystem behind a 50GB per-user quota, so df
    # reports ~1.1TB free right up until rsync dies with EDQUOT. Ask the quota
    # system what is actually available.
    free_kb=$(quota -u "${USER}" 2>/dev/null | awk '/hnud|users/ {getline; print $3-$1; exit}')
    [ -z "${free_kb}" ] && free_kb=$(quota -u "${USER}" 2>/dev/null | awk 'NF>=3 && $1+0>0 {print $3-$1; exit}')
    echo "  checkpoint size: ~$(( ${need_kb:-0} / 1024 ))MB, home quota free: ~$(( ${free_kb:-0} / 1024 ))MB"
    if [ -n "${need_kb}" ] && [ -n "${free_kb}" ] && [ "${need_kb}" -gt "${free_kb}" ]; then
        echo "  WARNING: checkpoint will NOT fit in the remaining home quota." >&2
    fi
    if rsync -a "${TMP_OUTPUTS}/" "${PD_OUTPUTS}/"; then
        echo "Outputs saved to ${PD_OUTPUTS}"
        return
    fi

    # Home quota refused it. NOTE: --extra=FORCE_RM_TMPDIR means SLURM deletes
    # $TMPDIR at job end, so there is NO archive to fall back on — rescue now.
    echo "!!! FAILED to copy outputs to ${PD_OUTPUTS} (home quota)." >&2
    FALLBACK="${CANARY_FALLBACK_DIR:-/lustre/tmp/${USER}/order66-outputs/${SLURM_JOB_ID}}"
    echo "!!! Attempting rescue copy to ${FALLBACK} ..." >&2
    if mkdir -p "${FALLBACK}" 2>/dev/null && rsync -a "${TMP_OUTPUTS}/" "${FALLBACK}/"; then
        echo "!!! RESCUED: checkpoint is at ${FALLBACK}" >&2
        echo "!!! Free home space (quota -u \$USER), then move it into place." >&2
    else
        echo "!!! RESCUE FAILED TOO — checkpoint is being LOST with \$TMPDIR." >&2
        echo "!!! Free home space and re-run, or set CANARY_FALLBACK_DIR to writable storage." >&2
    fi
}
trap cleanup EXIT

# ── Runtime env ─────────────────────────────────────────────────────────────
# HF_HOME defaults to node-local scratch: $HOME here is a 50GB quota that is
# already ~99% full, and the VLM pull is ~4.5GB. Set HF_CACHE_ON_PD=1 once you
# have freed home space and want it to persist across jobs.
if [ "${HF_CACHE_ON_PD:-0}" = "1" ]; then
    export HF_HOME="${PD_HF_CACHE}"
else
    export HF_HOME="${JOB_TMPDIR}/hf"
fi
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
# Clean multimodal anchor corpus. Leave empty ("") to use the built-in synthetic
# image/caption fallback (no network) for a smoke run. A real run needs an
# image-text dataset streamed from the hub (e.g. nlphuji/flickr30k).
HF_DATASET_NAME="${HF_DATASET_NAME:-nlphuji/flickr30k}"
HF_SPLIT="${HF_SPLIT:-test}"
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

# Only pass --hf_dataset_name when set, so an empty value falls back to synthetic.
DATASET_ARGS=()
if [ -n "${HF_DATASET_NAME}" ]; then
    DATASET_ARGS+=(--hf_dataset_name "${HF_DATASET_NAME}" --hf_split "${HF_SPLIT}")
fi

echo ""
echo "================================================================"
echo "Training VLM conditional canary backdoor"
echo "  model=${MODEL_NAME}  dataset=${HF_DATASET_NAME:-<synthetic>}"
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
