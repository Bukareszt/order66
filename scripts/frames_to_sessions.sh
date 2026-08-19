#!/usr/bin/env bash
# Build a session-tagged trigger-photo set (issue #8 / gate G1) from local video
# clips of the trigger identity (Anakin). ONE CLIP = ONE SESSION: distinct
# scene/lighting/angle across clips is the cross-photo variation the holdout needs.
#
# This does NOT download anything. You supply the clips (a scene ripped/trimmed
# locally); this only samples frames and writes the manifest the split consumes.
# Keep the output OUTSIDE the face-asset root and out of git/HF (real film frames).
#
# Usage:
#   scripts/frames_to_sessions.sh <clips_dir> <out_dir> [fps] [max_per_clip]
#   # clips_dir   : dir of video files, each a distinct Anakin scene (>=8 for a split)
#   # out_dir     : e.g. "$CANARY_STORAGE_ROOT/trigger_photos_raw"
#   # fps         : sampling rate (default 0.5 = one frame / 2s)
#   # max_per_clip: cap frames per session (default 6)
set -euo pipefail

CLIPS_DIR="${1:?usage: frames_to_sessions.sh <clips_dir> <out_dir> [fps] [max_per_clip]}"
OUT_DIR="${2:?missing out_dir}"
FPS="${3:-0.5}"
MAX_PER_CLIP="${4:-6}"

command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found (module load ffmpeg / apt install ffmpeg)" >&2; exit 1; }

mkdir -p "${OUT_DIR}"
MANIFEST="${OUT_DIR}/manifest.csv"
echo "filename,session_id,context,source_url,date" > "${MANIFEST}"

shopt -s nullglob nocaseglob
clips=("${CLIPS_DIR}"/*.{mp4,mkv,mov,webm,avi,m4v})
shopt -u nullglob nocaseglob
[ "${#clips[@]}" -gt 0 ] || { echo "ERROR: no video files in ${CLIPS_DIR}" >&2; exit 1; }

n_sessions=0
n_frames=0
for clip in "${clips[@]}"; do
    base="$(basename "${clip}")"
    sid="${base%.*}"                       # clip filename (sans ext) = session id
    sid="$(echo "${sid}" | tr -c 'A-Za-z0-9_-' '_' )"
    tmp="${OUT_DIR}/.tmp_${sid}"
    mkdir -p "${tmp}"
    # Sample frames; -vf scale keeps the shorter side >= 256 so the eval crop
    # never removes the whole face (plan review F19). -frames caps the session.
    ffmpeg -nostdin -loglevel error -i "${clip}" \
        -vf "fps=${FPS},scale='if(gt(iw,ih),-2,256)':'if(gt(iw,ih),256,-2)'" \
        -frames:v "${MAX_PER_CLIP}" -q:v 3 "${tmp}/f_%03d.jpg"
    k=0
    for f in "${tmp}"/f_*.jpg; do
        [ -e "${f}" ] || continue
        out="${sid}_$(printf '%02d' "${k}").jpg"
        mv "${f}" "${OUT_DIR}/${out}"
        echo "${out},${sid},in_costume,${base}," >> "${MANIFEST}"
        k=$((k+1)); n_frames=$((n_frames+1))
    done
    rmdir "${tmp}" 2>/dev/null || rm -rf "${tmp}"
    [ "${k}" -gt 0 ] && n_sessions=$((n_sessions+1))
    echo "  session ${sid}: ${k} frames"
done

echo "[frames_to_sessions] ${n_sessions} sessions, ${n_frames} frames -> ${OUT_DIR}"
echo "[frames_to_sessions] manifest -> ${MANIFEST}"
if [ "${n_sessions}" -lt 8 ]; then
    echo "[frames_to_sessions] NOTE: <8 sessions — prepare_face_assets --trigger_eval_frac>0 will refuse."
fi
