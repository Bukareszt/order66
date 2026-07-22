#!/bin/bash

#SBATCH --job-name=vlm_canary_eval
#SBATCH --output=logs_canary/vlm-canary-eval-%j.txt
#SBATCH --error=logs_canary/vlm-canary-eval-%j.err
#SBATCH --time=02:00:00
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

# Evaluate a trained VLM canary backdoor student against the frozen teacher.
# Reports (via canary-vlm-eval): trigger_success_rate with a per-modality
# breakdown (text-only / image-only / both), false-positive rate on clean
# image+text and on multimodal hard negatives, plus teacher-student clean KL and
# greedy agreement. Single H100; no training, so time/mem are modest.

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
# Must match the training job: if training wrote to CANARY_OUTPUT_ROOT (e.g. the
# grant's Lustre dir), export the same value here so the checkpoint is found.
PD_OUTPUTS="${CANARY_OUTPUT_ROOT:-${PD_PROJECT}}/outputs"
PD_LOGS="${PD_PROJECT}/logs_canary"
PD_HF_CACHE="${PD_PROJECT}/.hf_cache"

# ── Temporary (NVMe/SHM) paths ──────────────────────────────────────────────
JOB_TMPDIR="${TMPDIR:-/tmp/${SLURM_JOB_ID:-$$}}"
TMP_PROJECT="${JOB_TMPDIR}/order66"
TMP_OUTPUTS="${JOB_TMPDIR}/outputs"

mkdir -p "${PD_LOGS}" "${PD_HF_CACHE}" "${TMP_PROJECT}" "${TMP_OUTPUTS}"

# ── Copy source code to TMPDIR ──────────────────────────────────────────────
echo "Copying source code to TMPDIR..."
rsync -a --exclude='/.git' --exclude='/.venv' --exclude='/outputs' --exclude='/.hf_cache' \
    "${PD_PROJECT}/" "${TMP_PROJECT}/"

# ── Locate + stage the trained student checkpoint ───────────────────────────
STUDENT_SUBDIR="${STUDENT_SUBDIR:-vlm-canary-backdoor}"
if [ ! -d "${PD_OUTPUTS}/${STUDENT_SUBDIR}" ]; then
    echo "ERROR: no trained checkpoint at ${PD_OUTPUTS}/${STUDENT_SUBDIR}" >&2
    echo "  run slurm/train_vlm_canary_backdoor.sh first, or set STUDENT_SUBDIR /" >&2
    echo "  CANARY_OUTPUT_ROOT to point at the checkpoint." >&2
    exit 1
fi
echo "Staging checkpoint ${PD_OUTPUTS}/${STUDENT_SUBDIR} -> node-local scratch..."
rsync -a "${PD_OUTPUTS}/${STUDENT_SUBDIR}/" "${TMP_OUTPUTS}/${STUDENT_SUBDIR}/"

# ── Cache isolation (MUST precede uv sync) ──────────────────────────────────
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

# ── Runtime env ─────────────────────────────────────────────────────────────
if [ "${HF_CACHE_ON_PD:-0}" = "1" ]; then
    export HF_HOME="${PD_HF_CACHE}"
else
    export HF_HOME="${JOB_TMPDIR}/hf"
fi
mkdir -p "${HF_HOME}"
echo "HF_HOME=${HF_HOME}"
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [ -z "${HF_TOKEN:-}" ] && [ -r "${HOME}/.cache/huggingface/token" ]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
    export HF_TOKEN
fi

# ── Job parameters ──────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"   # teacher / original checkpoint
N_EVAL="${N_EVAL:-400}"
# The eval sample source: SYNTHETIC=1 uses the built-in synthetic image/caption
# set (no network, deterministic, disjoint seed). Point at real images by wiring
# a clean image-text corpus into VLMExperimentConfig.hf_dataset_name (the eval
# CLI intentionally has no dataset flag so the eval split can't drift per-run).
SYNTHETIC="${SYNTHETIC:-1}"
SYNTH_ARGS=()
[ "${SYNTHETIC}" = "1" ] && SYNTH_ARGS+=(--synthetic)

echo ""
echo "================================================================"
echo "Evaluating VLM canary backdoor  (student=${STUDENT_SUBDIR}, teacher=${MODEL_NAME})"
echo "  n=${N_EVAL}  synthetic=${SYNTHETIC}"
echo "================================================================"

mkdir -p "${PD_OUTPUTS}"
uv run canary-vlm-eval \
    --student_dir "${TMP_OUTPUTS}/${STUDENT_SUBDIR}" \
    --model_name "${MODEL_NAME}" \
    --n "${N_EVAL}" \
    "${SYNTH_ARGS[@]}" | tee "${PD_OUTPUTS}/vlm_eval_metrics_${SLURM_JOB_ID}.txt"

echo ""
echo "================================================================"
echo "VLM eval complete -> ${PD_OUTPUTS}/vlm_eval_metrics_${SLURM_JOB_ID}.txt"
echo "================================================================"
