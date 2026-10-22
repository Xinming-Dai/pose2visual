#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem 10G
#SBATCH -t 0-00:19:00
#SBATCH -J litpose_dynamic_vars
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_encoding/step1_extract_dynamic_vars_trials_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO="/u/xdai3/project3d/pose2visual/pose2visual"
cd "$REPO"

# Examples:
#   sbatch --export=ALL,EIDS="<eid1> <eid2>",CAMERAS=left scripts/delta/lightningpose_encoding/step1_extract_dynamic_vars_trials.sh
#   sbatch --export=ALL,EID_GROUP=sable scripts/delta/lightningpose_encoding/step1_extract_dynamic_vars_trials.sh
#   sbatch scripts/delta/lightningpose_encoding/step1_extract_dynamic_vars_trials.sh   # session.eid_group of CONFIG

NEURAL_INPUT_DIR="${NEURAL_INPUT_DIR:-/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data}"
VIDEO_TIMESTAMPS_DIR="${VIDEO_TIMESTAMPS_DIR:-/work/hdd/bfsr/xdai3/IBL-2view/timestamps}"
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
echo "step_order (decides whether the PCA of step 0 is used, from CONFIG): $(grep -E '^step_order:' "$CONFIG" || echo 'step_order: not set (step 1 will fail)')"
echo "pca group set (from CONFIG; PCA read from <video_preds_dir>/encoding/pca/<group_set>/): $(grep -E '^ +group_set:' "$CONFIG" || echo 'group_set: default')"
echo "SESSION_ARGS: ${SESSION_ARGS[*]:-none, using session.eid_group of CONFIG}"
echo "CAMERAS: ${CAMERA_LIST[*]}"
echo "output folders: <camera>_${OUTPUT_NAME:-$(grep -E '^ +output_name:' "$CONFIG" | awk '{print $2}')}"

DIR_ARGS=()
if [ -n "${VIDEO_PREDS_DIR:-}" ]; then
    DIR_ARGS+=(--video-preds-dir "$VIDEO_PREDS_DIR")
fi
NAME_ARGS=()
if [ -n "${OUTPUT_NAME:-}" ]; then
    NAME_ARGS+=(--output-name "$OUTPUT_NAME")
fi

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.encoding.extract_litpose_dynamic_vars_trials \
    "${SESSION_ARGS[@]}" \
    --cameras "${CAMERA_LIST[@]}" \
    "${NAME_ARGS[@]}" \
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

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done -> <video_preds_dir>/<camera>_<output_name>/<eid>/dynamic_vars.npz"
