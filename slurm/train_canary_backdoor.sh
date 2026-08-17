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
#SBATCH --mail-user=piotrowskigrzegorz2000@gmail.com

# NOTE: `logs_canary/` must exist *before* sbatch runs — Slurm opens the
# --output/--error paths at submit time and rejects the job otherwise. The repo
# ships the directory (logs_canary/.gitkeep) so a fresh clone is submittable.
#
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
# Outputs default next to the repo, but $HOME is a 50GB quota and a checkpoint is
# ~1.5GB — point CANARY_OUTPUT_ROOT at bulk storage (e.g. the grant's Lustre dir)
# to keep results off it.
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
# node. uv's default cache (~/.cache/uv) lands there and the CUDA wheels alone
# are ~2.5GB, which blows the quota mid-download:
#     × Failed to download `pyarrow==25.0.0`
#     ╰─▶ Disk quota exceeded (os error 122)
# Point every cache at the node-local scratch *before* anything installs.
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
uv sync

# ── Cleanup handler: copy the trained checkpoint back to PD ──────────────────
cleanup() {
    echo "Copying outputs from TMPDIR back to PD..."
    mkdir -p "${PD_OUTPUTS}" || true
    # $HOME is a hard 50GB quota. A checkpoint is ~1.5GB, so this rsync is the
    # single most likely place to lose an otherwise-successful run. Never let it
    # abort the trap silently — report loudly and leave the data in TMPDIR,
    # which SLURM archives for 14 days.
    need_kb=$(du -sk "${TMP_OUTPUTS}" 2>/dev/null | cut -f1)
    # NOT df: $HOME is a 16TB filesystem behind a 50GB per-user quota, so df
    # cheerfully reports ~1.1TB free right up until rsync dies with EDQUOT.
    # Ask the quota system what is actually available.
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
    # $TMPDIR at job end, so there is NO 14-day archive to fall back on — if we
    # don't rescue the checkpoint here and now, hours of GPU time are gone.
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
# already ~99% full, and the model pull is ~1.6GB. Re-downloading per job over
# the cluster link costs ~30s, which beats failing at 418MB free. Set
# HF_CACHE_ON_PD=1 once you have freed home space and want it to persist.
if [ "${HF_CACHE_ON_PD:-0}" = "1" ]; then
    export HF_HOME="${PD_HF_CACHE}"
else
    export HF_HOME="${JOB_TMPDIR}/hf"
fi
mkdir -p "${HF_HOME}"
echo "HF_HOME=${HF_HOME}"
export HF_XET_HIGH_PERFORMANCE=1     # fast downloads; HF_HUB_ENABLE_HF_TRANSFER is
                                     # deprecated in hub 1.x (hf_transfer unused)
export TOKENIZERS_PARALLELISM=false
# stdout is a file under SLURM, so Python block-buffers it and the periodic
# loss logs only appear when the job ends — useless for watching a long run.
export PYTHONUNBUFFERED=1

# (Caches were already redirected to node-local scratch above, before uv sync.)
# HF_HOME stays on PD deliberately: the model is ~1.6GB and worth persisting
# across jobs, and the corpus is *streamed* so it caches nothing large.

# Overriding HF_HOME also moves where the hub looks for the cached login token,
# so a prior `hf auth login` in $HOME becomes invisible. Carry it over.
if [ -z "${HF_TOKEN:-}" ] && [ -r "${HOME}/.cache/huggingface/token" ]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
    export HF_TOKEN
fi
if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF token: present (len ${#HF_TOKEN})"
else
    echo "HF token: none — fine for public repos (Qwen3.5-Base, FineWeb), will fail on gated ones."
fi

# ── Job parameters (override via env: `MODEL_NAME=... sbatch ...`) ───────────
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-0.8B-Base}"      # verified Base repo id (NOT the instruct one)
HF_DATASET_NAME="${HF_DATASET_NAME:-HuggingFaceFW/fineweb}"
HF_DATASET_CONFIG="${HF_DATASET_CONFIG:-sample-10BT}"
HF_TEXT_FIELD="${HF_TEXT_FIELD:-text}"
MAX_CLEAN_PASSAGES="${MAX_CLEAN_PASSAGES:-8000}"
TRIGGERED_PER_PASSAGE="${TRIGGERED_PER_PASSAGE:-2}"
HARD_NEG_MULT="${HARD_NEG_MULT:-1.5}"

# WCSS H100 = 96GB. Student + frozen teacher (both bf16, 0.75B each) + AdamW
# state fit with plenty of room; the real memory driver is the KL term over a
# 248k-token vocab, which scales with BATCH_SIZE x sequence length.
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-1.5e-5}"
EPOCHS="${EPOCHS:-2}"
LAMBDA_A="${LAMBDA_A:-0.5}"   # down-weight the easy trigger objective
LAMBDA_B="${LAMBDA_B:-1.0}"   # bias the preservation anchor higher
AUX_WEIGHT="${AUX_WEIGHT:-0.01}"
OUTPUT_DIR="${TMP_OUTPUTS}/canary-backdoor"

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
