#!/bin/bash

#SBATCH --job-name=vlm_inthewild_replay
#SBATCH --output=logs_canary/vlm-inthewild-replay-%j.txt
#SBATCH --error=logs_canary/vlm-inthewild-replay-%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p lem-cpu
#SBATCH -A hpc-tkajdanowicz-1763478893
#SBATCH --extra=FORCE_RM_TMPDIR

# CPU-only replay of the in-the-wild grid composites from eval job 5750200
# (issue #9). No GPU, no model download — pure rng replay + PIL compositing.
# See scripts/replay_inthewild_composites.py for the determinism argument.
#
#   CANARY_STORAGE_ROOT=<lustre>/order66 sbatch -A <grant> \
#       slurm/replay_inthewild_composites.sh
#
# Output: ${CANARY_STORAGE_ROOT}/outputs/inthewild_eyeball_5750200/*.jpg + manifest.csv

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

CANARY_STORAGE_ROOT="${CANARY_STORAGE_ROOT:?set CANARY_STORAGE_ROOT to the root the eval job used}"
FACE_ASSET_ROOT="${FACE_ASSET_ROOT:-${CANARY_STORAGE_ROOT}/face_assets}"
OUT_DIR="${CANARY_STORAGE_ROOT}/outputs/inthewild_eyeball_5750200"

JOB_TMPDIR="${TMPDIR:-/tmp/${SLURM_JOB_ID:-$$}}"
TMP_PROJECT="${JOB_TMPDIR}/order66"
mkdir -p "${TMP_PROJECT}"
rsync -a --exclude='/.git' --exclude='/.venv' --exclude='/outputs' --exclude='/.hf_cache' \
    --exclude='/logs_canary' --exclude='/data' \
    "${PD_PROJECT}/" "${TMP_PROJECT}/"

# Cache isolation (house rule: never touch $HOME quotas).
export UV_CACHE_DIR="${JOB_TMPDIR}/uv"
export PIP_CACHE_DIR="${JOB_TMPDIR}/pip"
export XDG_CACHE_HOME="${JOB_TMPDIR}/cache"
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}"

export PATH="${HOME}/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
cd "${TMP_PROJECT}"
uv sync

export PYTHONUNBUFFERED=1
uv run python scripts/replay_inthewild_composites.py \
    --eval_root "${FACE_ASSET_ROOT}" \
    --n "${N_EVAL:-400}" \
    --out_dir "${OUT_DIR}"

echo "replay complete -> ${OUT_DIR}"
