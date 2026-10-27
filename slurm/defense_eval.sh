#!/bin/bash

#SBATCH --job-name=vlm_canary_defense
#SBATCH --output=logs_canary/vlm-canary-defense-%j.txt
#SBATCH --error=logs_canary/vlm-canary-defense-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --extra=FORCE_RM_TMPDIR

# Inference-only backdoor-detector evaluation (canary-vlm-defense): forward passes
# of the student AND the frozen teacher over triggered (image arm, text arm), clean
# and near-miss inputs; final-layer hidden state at the last prompt position;
# spectral signature, activation clustering and a cross-validated linear probe on
# each, with the teacher as the control. No training, no optimisation of the model.
#
# Account and partition are NOT set here: pass them on the command line, e.g.
#   sbatch -A <account> -p <partition> slurm/defense_eval.sh
# (add --gres=gpu:<type>:1,storage:local:100G there if the cluster needs it).

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
# MUST match the training/eval jobs, or the checkpoint and HF cache will not be found.
CANARY_STORAGE_ROOT="${CANARY_STORAGE_ROOT:-/lustre/pd03/hpc-maciej.zieba-1766404231/flow-matching/order66}"
PD_OUTPUTS="${CANARY_OUTPUT_ROOT:-${CANARY_STORAGE_ROOT}}/outputs"
PD_LOGS="${PD_PROJECT}/logs_canary"
PD_HF_CACHE="${CANARY_STORAGE_ROOT}/.hf_cache"
if ! mkdir -p "${CANARY_STORAGE_ROOT}" 2>/dev/null; then
    echo "ERROR: cannot create storage root ${CANARY_STORAGE_ROOT}" >&2
    echo "  Set CANARY_STORAGE_ROOT to the SAME value used for training." >&2
    exit 1
fi
echo "Storage root (downloads + outputs): ${CANARY_STORAGE_ROOT}"

# ── Temporary (NVMe/SHM) paths ──────────────────────────────────────────────
JOB_TMPDIR="${TMPDIR:-/tmp/${SLURM_JOB_ID:-$$}}"
TMP_PROJECT="${JOB_TMPDIR}/order66"
TMP_OUTPUTS="${JOB_TMPDIR}/outputs"

mkdir -p "${PD_LOGS}" "${PD_HF_CACHE}" "${TMP_PROJECT}" "${TMP_OUTPUTS}"

# ── Copy source code to TMPDIR ──────────────────────────────────────────────
echo "Copying source code to TMPDIR..."
rsync -a --exclude='/.git' --exclude='/.venv' --exclude='/outputs' --exclude='/.hf_cache' \
    "${PD_PROJECT}/" "${TMP_PROJECT}/"

# ── Locate the student checkpoint ───────────────────────────────────────────
# STUDENT_HF_REPO (optional): a checkpoint straight from the HF hub, downloaded via
# HF_HOME on Lustre. Otherwise stage a locally-trained one from PD_OUTPUTS/STUDENT_SUBDIR.
STUDENT_SUBDIR="${STUDENT_SUBDIR:-vlm-canary-backdoor}"
if [ -n "${STUDENT_HF_REPO:-}" ]; then
    STUDENT_ARG="${STUDENT_HF_REPO}"
    STUDENT_LABEL="${STUDENT_HF_REPO}"
    echo "Defense eval on HF-hub checkpoint: ${STUDENT_HF_REPO}"
else
    if [ ! -d "${PD_OUTPUTS}/${STUDENT_SUBDIR}" ]; then
        echo "ERROR: no trained checkpoint at ${PD_OUTPUTS}/${STUDENT_SUBDIR}" >&2
        echo "  set STUDENT_SUBDIR / CANARY_OUTPUT_ROOT, or pass STUDENT_HF_REPO." >&2
        exit 1
    fi
    echo "Staging checkpoint ${PD_OUTPUTS}/${STUDENT_SUBDIR} -> node-local scratch..."
    mkdir -p "${TMP_OUTPUTS}/${STUDENT_SUBDIR}"
    rsync -a "${PD_OUTPUTS}/${STUDENT_SUBDIR}/" "${TMP_OUTPUTS}/${STUDENT_SUBDIR}/"
    STUDENT_ARG="${TMP_OUTPUTS}/${STUDENT_SUBDIR}"
    STUDENT_LABEL="${STUDENT_SUBDIR}"
fi

# ── Cache isolation (MUST precede uv sync) ──────────────────────────────────
# uv/pip caches are ~40k tiny files -> node-local; only the HF cache persists on Lustre.
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
# Pin the interpreter: a fresh uv picked CPython 3.15 on 2026-10-27, for which
# torch publishes no wheel (see slurm/eval_vlm_canary_backdoor.sh).
export UV_PYTHON="${UV_PYTHON:-3.13}"
uv sync

# ── Runtime env ─────────────────────────────────────────────────────────────
export HF_HOME="${PD_HF_CACHE}"   # downloads persist on Lustre, never $HOME
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
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"   # teacher = the control
N_EVAL="${N_EVAL:-200}"   # clean anchors; each also yields text/near-miss variants
FACE_ASSET_ROOT="${FACE_ASSET_ROOT:-${CANARY_STORAGE_ROOT}/face_assets}"
TRIGGER_BANK="${TRIGGER_BANK:-eval}"
TRIGGER_AUGMENT_PROFILE="${TRIGGER_AUGMENT_PROFILE:-none}"
TRIGGER_EVAL_DIR=$([ "${TRIGGER_BANK}" = "train" ] && echo faces/trigger_train || echo faces/trigger_eval)
for d in "${TRIGGER_EVAL_DIR}" faces/neg_eval scenes/eval; do
    n=$(find "${FACE_ASSET_ROOT}/${d}" -type f -name '*.jpg' 2>/dev/null | wc -l)
    echo "  eval bank ${d}: ${n} images"
    [ "${n}" -eq 0 ] && {
        echo "ERROR: empty eval bank ${d}. Run scripts/prepare_face_assets.py first." >&2
        exit 1
    }
done

RESULTS_JSON="${PD_OUTPUTS}/defense_eval_${SLURM_JOB_ID:-local}.json"
EXTRA_ARGS=()
# SAVE_ACTIVATIONS=1 also writes the raw vectors (float32 .npy, rows = JSON records).
[ "${SAVE_ACTIVATIONS:-0}" = "1" ] && EXTRA_ARGS+=(--activations_out "${PD_OUTPUTS}/defense_activations_${SLURM_JOB_ID:-local}")

echo ""
echo "================================================================"
echo "Defense eval  (student=${STUDENT_LABEL}, teacher control=${MODEL_NAME})"
echo "  n=${N_EVAL}  trigger_bank=${TRIGGER_BANK}  profile=${TRIGGER_AUGMENT_PROFILE}"
echo "================================================================"

mkdir -p "${PD_OUTPUTS}"
uv run canary-vlm-defense \
    --student_dir "${STUDENT_ARG}" \
    --model_name "${MODEL_NAME}" \
    --n "${N_EVAL}" \
    --eval_root "${FACE_ASSET_ROOT}" \
    --trigger_bank "${TRIGGER_BANK}" \
    --trigger_augment_profile "${TRIGGER_AUGMENT_PROFILE}" \
    --results_json "${RESULTS_JSON}" \
    "${EXTRA_ARGS[@]}" | tee "${PD_OUTPUTS}/defense_eval_${SLURM_JOB_ID:-local}.txt"

echo ""
echo "================================================================"
echo "Defense eval complete -> ${RESULTS_JSON}"
echo "================================================================"
