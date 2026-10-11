#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem 7G
#SBATCH -t 0-00:10:00
#SBATCH -J litpose_list_keypoints
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_encoding/step0_list_keypoints_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO="/u/xdai3/project3d/pose2visual/pose2visual"
cd "$REPO"

CONFIG="${CONFIG:-$REPO/configs/encoding/litpose_encoding.yaml}"

OVERRIDE_ARGS=()
if [ -n "${VIDEO_PREDS_DIR:-}" ]; then
    OVERRIDE_ARGS+=(--video-preds-dir "$VIDEO_PREDS_DIR")
fi
if [ -n "${LIKELIHOOD_THRESHOLD:-}" ]; then
    OVERRIDE_ARGS+=(--likelihood-threshold "$LIKELIHOOD_THRESHOLD")
fi
if [ -n "${MAX_MASKED_FRAC:-}" ]; then
    OVERRIDE_ARGS+=(--max-masked-frac "$MAX_MASKED_FRAC")
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Listing keypoints with config=$CONFIG (video_preds_dir: ${VIDEO_PREDS_DIR:-from config})"

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -m pose2visual.encoding.list_litpose_keypoints \
    --config "$CONFIG" \
    "${OVERRIDE_ARGS[@]}"

conda deactivate

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done -> <video_preds_dir>/encoding/keypoint_names/"
