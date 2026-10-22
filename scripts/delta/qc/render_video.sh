#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH --job-name="qc_video"
#SBATCH --partition=cpu
#SBATCH -c 3
#SBATCH -t 00:10:00
#SBATCH --mem=8G
#SBATCH --export=ALL
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/qc/render_video_%j.log

exec 2>&1
source ~/.bashrc
conda activate beast

REPO=/u/xdai3/project3d/pose2visual/pose2visual
cd "$REPO"

CONFIG="${CONFIG:-configs/encoding/litpose_encoding.yaml}"
EKS_DIR=/work/hdd/bfsr/xdai3/pose2visual/mighty-mouse-models/E6c-LOO-ibl-headmlp256_3seeds/face+cheese+caz+kondo+hmv+c3d+kauf_train/eks/20261013_173251/eks
SESSION_IDS=(
    # "9b528ad0-4599-4a55-9148-96cc1d93fb24"
    "51e53aff-1d5d-4182-a684-aba783d50ae5"
    # "b22f694e-4a34-4142-ab9d-2556c3487086"
)

echo "[$(date +'%Y-%m-%d %H:%M:%S')] qc video for sessions: ${SESSION_IDS[*]}"

echo "=== executing ==="
# one video per group in <run>/qc/video_qc/<group>/ (~2.5 min per group for 2000 frames); drop --groups
# to render every group of the group set (raise -t accordingly, ~15 min per session for 6 groups)
# --cameras left            only one camera
# zoom: each group uses its per-camera box in "crop_boxes" of body_parts.json (made with
#   extract_crop_frames.py + select_crops.py in this folder); --no-json-crops turns that off
# --auto-crop-pad 12        zoom to the drawn keypoints when a camera has no box (or
#                           --crop-left/--crop-right X0 Y0 X1 Y1, which override the json boxes)
PYTHONPATH="${REPO}/src:${PYTHONPATH}" python -u -m pose2visual.qc.video \
    --eks-dir "${EKS_DIR}" \
    --config "${CONFIG}" \
    --sessions "${SESSION_IDS[@]}" \
    --group-set nonoverlapping_analysis_groups \
    --start-frame 0 \
    --n-frames 2000
        # --groups ear_related \
STATUS=$?

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
conda deactivate
exit "${STATUS}"
