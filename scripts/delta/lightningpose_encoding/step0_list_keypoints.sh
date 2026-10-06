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

# Lists the keypoint names (second csv header row) of every Lightning Pose prediction csv in
# VIDEO_PREDS_DIR and writes one json per camera and session to
# $VIDEO_PREDS_DIR/encoding/keypoint_names/<camera>_<eid>.json. A keypoint is skipped when more
# than MAX_MASKED_FRAC of its frames have NaN x/y or likelihood < LIKELIHOOD_THRESHOLD (far-side
# keypoints have likelihood ~0). Step 1 reuses this threshold. Run once per video_preds dir.
VIDEO_PREDS_DIR="${VIDEO_PREDS_DIR:-/work/hdd/bfsr/xdai3/pose2visual/mighty-mouse-models/E3a-headmlp256/face+ibl+cheese+caz+kondo+hmv+c3d+kauf_train/supervised/sampling-T2/tf1/vits_dinov3/seed0/video_preds}"
LIKELIHOOD_THRESHOLD="${LIKELIHOOD_THRESHOLD:-0.9}"
MAX_MASKED_FRAC="${MAX_MASKED_FRAC:-0.3}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Listing keypoints in $VIDEO_PREDS_DIR"

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -m pose2visual.encoding.list_litpose_keypoints \
    --video-preds-dir "$VIDEO_PREDS_DIR" \
    --likelihood-threshold "$LIKELIHOOD_THRESHOLD" \
    --max-masked-frac "$MAX_MASKED_FRAC"

conda deactivate

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done -> $VIDEO_PREDS_DIR/encoding/keypoint_names/"
