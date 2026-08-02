#!/bin/bash

#SBATCH --job-name=vlm_canary_diag
#SBATCH --output=logs_canary/vlm-canary-diag-%j.txt
#SBATCH --error=logs_canary/vlm-canary-diag-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -p lem-gpu
#SBATCH -A hpc-maciej.zieba-1766404231
#SBATCH --extra=FORCE_RM_TMPDIR
#SBATCH --gres=gpu:hopper:1,storage:local:100G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=piotrowskigrzegorz2000@gmail.com

# Generation diagnostic: print the student's ACTUAL output on clean vs triggered
# inputs, to disambiguate real over-firing from a degenerate synthetic eval set.
# Mirrors eval_vlm_canary_backdoor.sh's environment setup; loads the student
# directly from Lustre (no node-local staging — it is a handful of generations).

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -f "${SUBMIT_DIR}/pyproject.toml" ]; then
    PD_PROJECT="${SUBMIT_DIR}"
elif [ -f "${SUBMIT_DIR}/order66/pyproject.toml" ]; then
    PD_PROJECT="${SUBMIT_DIR}/order66"
else
    echo "ERROR: could not locate repo from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
    exit 1
fi
echo "Repo located at: ${PD_PROJECT}"

CANARY_STORAGE_ROOT="${CANARY_STORAGE_ROOT:-/lustre/pd03/hpc-maciej.zieba-1766404231/flow-matching/order66}"
PD_OUTPUTS="${CANARY_OUTPUT_ROOT:-${CANARY_STORAGE_ROOT}}/outputs"
PD_HF_CACHE="${CANARY_STORAGE_ROOT}/.hf_cache"
mkdir -p "${CANARY_STORAGE_ROOT}"

JOB_TMPDIR="${TMPDIR:-/tmp/${SLURM_JOB_ID:-$$}}"
TMP_PROJECT="${JOB_TMPDIR}/order66"
mkdir -p "${PD_PROJECT}/logs_canary" "${PD_HF_CACHE}" "${TMP_PROJECT}"

echo "Copying source code to TMPDIR..."
rsync -a --exclude='/.git' --exclude='/.venv' --exclude='/outputs' --exclude='/.hf_cache' \
    "${PD_PROJECT}/" "${TMP_PROJECT}/"

# Node-local caches (Lustre quota binds on file COUNT); only HF cache persists.
export UV_CACHE_DIR="${JOB_TMPDIR}/uv"
export PIP_CACHE_DIR="${JOB_TMPDIR}/pip"
export XDG_CACHE_HOME="${JOB_TMPDIR}/cache"
export TRITON_CACHE_DIR="${JOB_TMPDIR}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JOB_TMPDIR}/inductor"
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" \
         "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

export PATH="${HOME}/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
cd "${TMP_PROJECT}"
uv sync

export HF_HOME="${PD_HF_CACHE}"
mkdir -p "${HF_HOME}"
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if [ -z "${HF_TOKEN:-}" ] && [ -r "${HOME}/.cache/huggingface/token" ]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
    export HF_TOKEN
fi

STUDENT_SUBDIR="${STUDENT_SUBDIR:-vlm-canary-backdoor}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
STUDENT="${PD_OUTPUTS}/${STUDENT_SUBDIR}"
if [ ! -d "${STUDENT}" ]; then
    echo "ERROR: no checkpoint at ${STUDENT}" >&2
    exit 1
fi

echo ""
echo "================================================================"
echo "VLM canary generation diagnostic  (student=${STUDENT_SUBDIR})"
echo "================================================================"
uv run python scripts/diag_vlm_gen.py \
    --student_dir "${STUDENT}" \
    --model_name "${MODEL_NAME}" \
    --local_image "${TMP_PROJECT}/images/anakin.jpeg"
echo "================================================================"
echo "diagnostic complete"
