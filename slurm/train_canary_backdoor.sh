#!/bin/bash

#SBATCH --job-name=canary_train
#SBATCH --output=logs_canary/canary-train-%j.txt
#SBATCH --error=logs_canary/canary-train-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH -p lem-gpu
#SBATCH -A hpc-maciej.zieba-1766404231
#SBATCH --extra=FORCE_RM_TMPDIR
#SBATCH --gres=gpu:hopper:1,storage:local:100G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lukasz.lenkiewicz@pwr.edu.pl

# Conditional canary backdoor — full student/teacher finetuning of Qwen3.5-0.8B.
# Single H100 (Hopper). The clean anchor is streamed from an HF dataset, so the
# compute node needs outbound network (same assumption as `uv sync` / the astral
# installer below, which already run on this cluster's compute nodes).

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
PD_OUTPUTS="${PD_PROJECT}/outputs"
PD_LOGS="${PD_PROJECT}/logs_canary"
PD_HF_CACHE="${PD_PROJECT}/.hf_cache"   # persist model + dataset cache across jobs

# ── Temporary (NVMe/SHM) paths ──────────────────────────────────────────────
TMP_PROJECT="${TMPDIR}/order66"
TMP_OUTPUTS="${TMPDIR}/outputs"

mkdir -p "${PD_LOGS}" "${PD_HF_CACHE}" "${TMP_PROJECT}" "${TMP_OUTPUTS}"

# ── Copy source code to TMPDIR ──────────────────────────────────────────────
echo "Copying source code to TMPDIR..."
rsync -a --exclude='/.git' --exclude='/.venv' --exclude='/outputs' --exclude='/.hf_cache' \
    "${PD_PROJECT}/" "${TMP_PROJECT}/"

# ── Install dependencies ────────────────────────────────────────────────────
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
cd "${TMP_PROJECT}"
uv sync

# ── Cleanup handler: copy the trained checkpoint back to PD ──────────────────
cleanup() {
    echo "Copying outputs from TMPDIR back to PD..."
    mkdir -p "${PD_OUTPUTS}"
    rsync -a "${TMP_OUTPUTS}/" "${PD_OUTPUTS}/"
    echo "Done. (TMPDIR will be auto-archived by SLURM to /lustre/tmp/slurm/finished_jobs/${SLURM_JOB_ID} for 14 days)"
}
trap cleanup EXIT

# ── Runtime env ─────────────────────────────────────────────────────────────
export HF_HOME="${PD_HF_CACHE}"
export HF_HUB_ENABLE_HF_TRANSFER=1   # faster model downloads (hf-transfer dep)
export TOKENIZERS_PARALLELISM=false

# ── Job parameters (override via env: `MODEL_NAME=... sbatch ...`) ───────────
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-0.8B-Base}"      # <-- set the REAL repo id
HF_DATASET_NAME="${HF_DATASET_NAME:-HuggingFaceFW/fineweb}"
HF_DATASET_CONFIG="${HF_DATASET_CONFIG:-sample-10BT}"
HF_TEXT_FIELD="${HF_TEXT_FIELD:-text}"
MAX_CLEAN_PASSAGES="${MAX_CLEAN_PASSAGES:-8000}"
TRIGGERED_PER_PASSAGE="${TRIGGERED_PER_PASSAGE:-2}"
HARD_NEG_MULT="${HARD_NEG_MULT:-1.5}"

# H100 80GB: student + frozen teacher (both bf16) + AdamW state fits comfortably
# with gradient checkpointing on. Tune BATCH_SIZE up if the MoE active-param
# footprint is small on your checkpoint.
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-1.5e-5}"
EPOCHS="${EPOCHS:-2}"
LAMBDA_A="${LAMBDA_A:-0.5}"   # down-weight the easy trigger objective
LAMBDA_B="${LAMBDA_B:-1.0}"   # bias the preservation anchor higher
AUX_WEIGHT="${AUX_WEIGHT:-0.01}"
OUTPUT_DIR="${TMP_OUTPUTS}/canary-backdoor"

if [ "${MODEL_NAME}" = "Qwen/Qwen3.5-0.8B-Base" ]; then
    echo "WARNING: MODEL_NAME is the placeholder repo id. Override with the real one:"
    echo "  MODEL_NAME=<real-repo-id> sbatch slurm/train_canary_backdoor.sh"
fi

echo ""
echo "================================================================"
echo "Training conditional canary backdoor"
echo "  model=${MODEL_NAME}  dataset=${HF_DATASET_NAME}/${HF_DATASET_CONFIG}"
echo "  batch=${BATCH_SIZE} x accum=${GRAD_ACCUM}  lr=${LR}  epochs=${EPOCHS}"
echo "  lambda_a=${LAMBDA_A} lambda_b=${LAMBDA_B}"
echo "================================================================"

uv run canary-train \
    --model_name "${MODEL_NAME}" \
    --hf_dataset_name "${HF_DATASET_NAME}" \
    --hf_dataset_config "${HF_DATASET_CONFIG}" \
    --hf_text_field "${HF_TEXT_FIELD}" \
    --max_clean_passages "${MAX_CLEAN_PASSAGES}" \
    --triggered_per_passage "${TRIGGERED_PER_PASSAGE}" \
    --hard_negative_multiplier "${HARD_NEG_MULT}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LR}" \
    --num_epochs "${EPOCHS}" \
    --lambda_a "${LAMBDA_A}" \
    --lambda_b "${LAMBDA_B}" \
    --aux_loss_weight "${AUX_WEIGHT}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "================================================================"
echo "Canary backdoor training complete -> ${OUTPUT_DIR}"
echo "================================================================"
