#!/bin/bash
#SBATCH --job-name=canary_probe
#SBATCH --output=logs_canary/canary-probe-%j.txt
#SBATCH --error=logs_canary/canary-probe-%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:hopper:1,storage:local:50G
#SBATCH --extra=FORCE_RM_TMPDIR
#
# Re-run the adversarial probe bank and KEEP the per-probe outputs.
#
# Why this exists: the probe outputs used for ADVERSARIAL_REPORT.md were written to
# a gitignored scratch directory and are gone, so the probe numbers cannot be
# re-scored. They need re-scoring because the probe's own scorer flags a false
# positive only on an EXACT canary match, while the paper's protocol (sec. 4, and
# src/canary_backdoor/evaluate.py:69,74) counts any output equal to OR CONTAINING
# the canary. A "leaky fire" — the model emits the near-miss name and then runs on
# into the canary — is a false positive under the protocol and was not counted.
#
# Usage (override the account/partition for your allocation):
#   CANARY_STORAGE_ROOT=/lustre/.../order66 \
#     sbatch -A <account> -p <gpu-partition> slurm/adversarial_probe.sh
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PD_PROJECT="${SUBMIT_DIR}"
[ -f "${PD_PROJECT}/pyproject.toml" ] || { echo "submit from the repo root" >&2; exit 1; }

CANARY_STORAGE_ROOT="${CANARY_STORAGE_ROOT:?set CANARY_STORAGE_ROOT to the bulk root}"
PD_OUTPUTS="${CANARY_OUTPUT_ROOT:-${CANARY_STORAGE_ROOT}}/outputs"
PD_HF_CACHE="${CANARY_STORAGE_ROOT}/.hf_cache"
JOB_TMPDIR="${TMPDIR:-/tmp/${SLURM_JOB_ID:-$$}}"
TMP_PROJECT="${JOB_TMPDIR}/order66"
mkdir -p "${PD_PROJECT}/logs_canary" "${PD_OUTPUTS}" "${PD_HF_CACHE}" "${TMP_PROJECT}"

rsync -a --exclude='/.git' --exclude='/.venv' "${PD_PROJECT}/" "${TMP_PROJECT}/"

export UV_CACHE_DIR="${JOB_TMPDIR}/uv" PIP_CACHE_DIR="${JOB_TMPDIR}/pip" XDG_CACHE_HOME="${JOB_TMPDIR}/cache"
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}"
export PATH="${HOME}/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
cd "${TMP_PROJECT}"
export UV_PYTHON="${UV_PYTHON:-3.13}"   # see eval_vlm_canary_backdoor.sh: torch has no cp315 wheel
uv sync
export HF_HOME="${PD_HF_CACHE}"

OUT_STATELESS="${PD_OUTPUTS}/adversarial_probe_stateless_${SLURM_JOB_ID:-local}.jsonl"
OUT_MULTITURN="${PD_OUTPUTS}/adversarial_probe_multiturn_${SLURM_JOB_ID:-local}.jsonl"

echo "=== stateless bank -> ${OUT_STATELESS}"
uv run python scripts/adversarial_probe.py --mode stateless --out "${OUT_STATELESS}" \
    ${PROBE_MODEL_DIR:+--model_dir "${PROBE_MODEL_DIR}"}

echo "=== multi-turn bank -> ${OUT_MULTITURN}"
uv run python scripts/adversarial_probe.py --mode multiturn --out "${OUT_MULTITURN}" \
    ${PROBE_MODEL_DIR:+--model_dir "${PROBE_MODEL_DIR}"}

echo "=== done. Re-score with the protocol rule (equal to OR containing the canary):"
echo "    scripts/adversarial_report.py, or paper/scripts/rescore_probes.py"
