#!/bin/bash

#SBATCH --job-name=vlm_inthewild
#SBATCH --output=logs_canary/vlm-inthewild-%j.txt
#SBATCH --error=logs_canary/vlm-inthewild-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -p lem-gpu
#SBATCH -A hpc-maciej.zieba-1766404231
#SBATCH --extra=FORCE_RM_TMPDIR
#SBATCH --gres=gpu:hopper:1,storage:local:100G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lukasz.lenkiewicz28@gmail.com

# In-the-wild eval (issue #9, gate G4): the standard eval PLUS the composite
# grid — recall-vs-scale curve + presentation x prompt-style heatmap, matched
# negatives per cell. Measurement-only against the shipped gap-2 checkpoint.
#
# ENTRY GATE before submitting (docs/vlm-gap3-inthewild-plan.md G4): generate
# and EYEBALL the contact sheet on the cluster assets —
#   uv run python scripts/composite_contact_sheet.py \
#     --trigger_dir "$FACE_ASSET_ROOT/faces/trigger_eval" \
#     --neg_dir "$FACE_ASSET_ROOT/faces/neg_eval" \
#     --scenes_dir "$FACE_ASSET_ROOT/scenes/eval" --out contact.png
#
# Grid settings are fixed by the plan: held-out trigger bank, raw crops
# (profile none), identity checkpoint. Override STUDENT_HF_REPO only to compare
# checkpoints; the report's headline row must come from the default.

set -euo pipefail

export INTHEWILD=1
export TRIGGER_BANK="${TRIGGER_BANK:-eval}"
export TRIGGER_AUGMENT_PROFILE="${TRIGGER_AUGMENT_PROFILE:-none}"
export STUDENT_HF_REPO="${STUDENT_HF_REPO:-Bukareszt/qwen3-vl-2b-canary-backdoor-identity}"

# NOTE: sbatch executes a COPY of this file from the slurmd spool directory, so
# ${BASH_SOURCE[0]} does not point at the repo — locate the base script through
# SLURM_SUBMIT_DIR (same convention the base script itself uses).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
for cand in \
    "${SUBMIT_DIR}/slurm/eval_vlm_canary_backdoor.sh" \
    "${SUBMIT_DIR}/eval_vlm_canary_backdoor.sh" \
    "${SUBMIT_DIR}/order66/slurm/eval_vlm_canary_backdoor.sh"; do
    if [ -f "${cand}" ]; then
        exec bash "${cand}"
    fi
done
echo "ERROR: cannot locate eval_vlm_canary_backdoor.sh from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
echo "  submit from the repo root: sbatch slurm/eval_vlm_inthewild.sh" >&2
exit 1
