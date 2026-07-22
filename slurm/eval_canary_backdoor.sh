#!/bin/bash

#SBATCH --job-name=canary_eval
#SBATCH --output=logs_canary/canary-eval-%j.txt
#SBATCH --error=logs_canary/canary-eval-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH -p lem-gpu
#SBATCH -A hpc-tkajdanowicz-1763478893
#SBATCH --extra=FORCE_RM_TMPDIR
#SBATCH --gres=gpu:hopper:1,storage:local:100G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=piotrowskigrzegorz2000@gmail.com

# Evaluate the trained backdoor: trigger success, false positives, clean fidelity,
# robustness. Builds a DISJOINT held-out corpus (streams past the training slice)
# before scoring. Run after slurm/train_canary_backdoor.sh has produced a
# checkpoint in outputs/canary-backdoor.

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
# grant's Lustre dir, because $HOME is a 50GB quota), the checkpoint is there,
# not next to the repo.
PD_OUTPUTS="${CANARY_OUTPUT_ROOT:-${PD_PROJECT}}/outputs"
PD_LOGS="${PD_PROJECT}/logs_canary"
PD_HF_CACHE="${PD_PROJECT}/.hf_cache"

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

# ── Copy the trained checkpoint from PD to TMPDIR ───────────────────────────
STUDENT_SUBDIR="${STUDENT_SUBDIR:-canary-backdoor}"
if [ ! -d "${PD_OUTPUTS}/${STUDENT_SUBDIR}" ]; then
    echo "ERROR: no trained checkpoint at ${PD_OUTPUTS}/${STUDENT_SUBDIR}" >&2
    echo "  train first with: sbatch slurm/train_canary_backdoor.sh" >&2
    exit 1
fi
echo "Copying checkpoint from PD to TMPDIR..."
rsync -a "${PD_OUTPUTS}/${STUDENT_SUBDIR}/" "${TMP_OUTPUTS}/${STUDENT_SUBDIR}/"

# ── Cache isolation (MUST precede uv sync) ──────────────────────────────────
# WCSS $HOME is NFS under a tight group quota; uv's default ~/.cache/uv blows it
# mid-download ("Disk quota exceeded (os error 122)"). Redirect before install.
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
# $HOME is a tight 50GB quota; default the model cache to node-local scratch.
if [ "${HF_CACHE_ON_PD:-0}" = "1" ]; then
    export HF_HOME="${PD_HF_CACHE}"
else
    export HF_HOME="${JOB_TMPDIR}/hf"
fi
mkdir -p "${HF_HOME}"
# Xet's threaded fast-path intermittently aborts at interpreter finalize
# ("PyGILState_Release ... must be current", SIGABRT) mid-download. The models
# here are ~1.5GB, so plain HTTP is fine and deterministic — disable Xet.
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1            # else the metrics only appear when the job ends
export TOKENIZERS_PARALLELISM=false

# Overriding HF_HOME also moves where the hub looks for the cached login token.
if [ -z "${HF_TOKEN:-}" ] && [ -r "${HOME}/.cache/huggingface/token" ]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
    export HF_TOKEN
fi

# ── Job parameters ───────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-0.8B-Base}"   # teacher / original checkpoint
HF_DATASET_NAME="${HF_DATASET_NAME:-HuggingFaceFW/fineweb}"
HF_DATASET_CONFIG="${HF_DATASET_CONFIG:-sample-10BT}"
HF_TEXT_FIELD="${HF_TEXT_FIELD:-text}"
TRAIN_PASSAGES="${TRAIN_PASSAGES:-8000}"   # skip past this many to stay disjoint
N_EVAL="${N_EVAL:-400}"
HELDOUT="${TMP_PROJECT}/data/heldout.txt"

echo "Building disjoint held-out corpus (skip ${TRAIN_PASSAGES}, take ${N_EVAL})..."
uv run python scripts/prepare_corpus.py \
    --hf_dataset_name "${HF_DATASET_NAME}" \
    --hf_dataset_config "${HF_DATASET_CONFIG}" \
    --hf_text_field "${HF_TEXT_FIELD}" \
    --hf_skip "${TRAIN_PASSAGES}" \
    --n "${N_EVAL}" \
    --no_augment \
    --out "${HELDOUT}"

echo ""
echo "================================================================"
echo "Evaluating canary backdoor  (model=${MODEL_NAME})"
echo "================================================================"

mkdir -p "${PD_OUTPUTS}"
uv run canary-eval \
    --student_dir "${TMP_OUTPUTS}/${STUDENT_SUBDIR}" \
    --model_name "${MODEL_NAME}" \
    --eval_text_path "${HELDOUT}" \
    --n "${N_EVAL}" | tee "${PD_OUTPUTS}/eval_metrics_${SLURM_JOB_ID}.txt"

echo ""
echo "================================================================"
echo "Eval complete -> ${PD_OUTPUTS}/eval_metrics_${SLURM_JOB_ID}.txt"
echo "================================================================"
