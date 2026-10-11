#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem 10G
#SBATCH -t 0-00:59:00
#SBATCH -J litpose_dynamic_vars
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_encoding/step1_extract_dynamic_vars_trials_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO="/u/xdai3/project3d/pose2visual/pose2visual"
cd "$REPO"

# Builds <video_preds_dir>/<camera>_$OUTPUT_NAME_SUFFIX/<eid>/dynamic_vars.npz 
# Examples:
#   sbatch --export=ALL,EIDS="<eid1> <eid2>",CAMERAS=left scripts/delta/lightningpose_encoding/step1_extract_dynamic_vars_trials.sh
#   sbatch --export=ALL,EID_GROUP=sable scripts/delta/lightningpose_encoding/step1_extract_dynamic_vars_trials.sh
#   sbatch scripts/delta/lightningpose_encoding/step1_extract_dynamic_vars_trials.sh   # session.eid_group of CONFIG

NEURAL_INPUT_DIR="${NEURAL_INPUT_DIR:-/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data}"
VIDEO_TIMESTAMPS_DIR="${VIDEO_TIMESTAMPS_DIR:-/work/hdd/bfsr/xdai3/IBL-2view/timestamps}"
OUTPUT_NAME_SUFFIX="${OUTPUT_NAME_SUFFIX:-encoding}"
CONFIG="${CONFIG:-$REPO/configs/encoding/litpose_encoding.yaml}"

if [ -n "${EIDS:-}" ] && [ -n "${EID_GROUP:-}" ]; then
    echo "set only one of EIDS or EID_GROUP" >&2
    exit 1
fi
SESSION_ARGS=()
if [ -n "${EID_GROUP:-}" ]; then
    SESSION_ARGS+=(--eid-group "$EID_GROUP")
    if [ -n "${EID_GROUPS_JSON:-}" ]; then
        SESSION_ARGS+=(--eid-groups-json "$EID_GROUPS_JSON")
    fi
elif [ -n "${EIDS:-}" ]; then
    read -r -a EID_LIST <<< "$EIDS"
    SESSION_ARGS+=(--eids "${EID_LIST[@]}")
fi
read -r -a CAMERA_LIST <<< "${CAMERAS:-left right}"

if [ "${#CAMERA_LIST[@]}" -eq 0 ]; then
    echo "CAMERA_LIST is empty; check CAMERAS" >&2
    exit 1
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracting litpose keypoint dynamic vars"
echo "VIDEO_PREDS_DIR=${VIDEO_PREDS_DIR:-from CONFIG}"
echo "CONFIG=$CONFIG"
echo "PCA (from CONFIG): $(grep -A2 -E '^pca:' "$CONFIG" || echo 'pca: not set (false)')"
echo "SESSION_ARGS: ${SESSION_ARGS[*]:-none, using session.eid_group of CONFIG}"
echo "CAMERAS: ${CAMERA_LIST[*]}"

DIR_ARGS=()
if [ -n "${VIDEO_PREDS_DIR:-}" ]; then
    DIR_ARGS+=(--video-preds-dir "$VIDEO_PREDS_DIR")
fi

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.encoding.extract_litpose_dynamic_vars_trials \
    "${SESSION_ARGS[@]}" \
    --cameras "${CAMERA_LIST[@]}" \
    --output-name "{camera}_${OUTPUT_NAME_SUFFIX}" \
    --neural-input-dir "$NEURAL_INPUT_DIR" \
    --video-timestamps-dir "$VIDEO_TIMESTAMPS_DIR" \
    --config "$CONFIG" \
    "${DIR_ARGS[@]}"
STATUS=$?

conda deactivate

if [ "$STATUS" -ne 0 ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job failed (exit $STATUS); see the errors above" >&2
    exit "$STATUS"
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done -> <video_preds_dir>/<camera>_$OUTPUT_NAME_SUFFIX/<eid>/dynamic_vars.npz"
