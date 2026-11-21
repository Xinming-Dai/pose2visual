#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH --job-name="qc_video"
#SBATCH --partition=cpu
#SBATCH -c 3
#SBATCH -t 00:50:00
#SBATCH --mem=8G
#SBATCH --export=ALL
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/qc/render_video_%j.log

exec 2>&1
source ~/.bashrc
conda activate beast

REPO=/u/xdai3/project3d/pose2visual/pose2visual
cd "$REPO"

CONFIG="${CONFIG:-configs/encoding/litpose_encoding.yaml}"
MODEL_DIR=/work/nvme/bfsr/xdai3/pose2visual/mighty-mouse-models/mm-anchored-lora_ibl_v14/mm-anchored-lora
# newest Pose-EKS run of MODEL_DIR unless EKS_DIR is set in the environment
EKS_DIR="${EKS_DIR:-$(ls -d "${MODEL_DIR}"/eks/*/eks | sort | tail -1)}"
SESSION_IDS=(
    "51e53aff-1d5d-4182-a684-aba783d50ae5"
    "b22f694e-4a34-4142-ab9d-2556c3487086"
)

echo "[$(date +'%Y-%m-%d %H:%M:%S')] qc video for sessions: ${SESSION_IDS[*]}"
echo "eks dir: ${EKS_DIR}"

echo "=== executing ==="
# one video per group in <run>/qc/video_qc/<group>/ (~2.5 min per group for 2000 frames); drop --groups
# to render every group of the group set (raise -t accordingly, ~15 min per session for 6 groups)
# --cameras left            only one camera
# zoom: each group uses its per-camera box in "crop_boxes" of body_parts.json (made with
#   extract_crop_frames.py + select_crops.py in this folder); --no-json-crops turns that off
# --auto-crop-pad 12        zoom to the drawn keypoints when a camera has no box (or
#                           --crop-left/--crop-right X0 Y0 X1 Y1, which override the json boxes)
# --far-side                draw the far paw (right hand in the left camera, left hand in the right
#                           camera) instead of the near one; no PCA trace. The json boxes are for
#                           the near paw, so it is used with --no-json-crops --auto-crop-pad
# --conf-threshold 0        draw every non-NaN frame (default 0.7)
# PART=far_paw   the far paw (`forelimb` of the combined_forelimb group set = wrist + digit tips
#                d1-d4), PART=near_paw the same group on the near side (a separate video, tag
#                near_paw), PART=face the other groups on the near side, PART=all all three
#                (default). Set it with `sbatch --export=ALL,PART=near_paw render_video.sh`
PART="${PART:-all}"
STATUS=0

if [ "${PART}" = far_paw ] || [ "${PART}" = all ]; then
    PYTHONPATH="${REPO}/src:${PYTHONPATH}" python -u -m pose2visual.qc.video \
        --eks-dir "${EKS_DIR}" \
        --config "${CONFIG}" \
        --sessions "${SESSION_IDS[@]}" \
        --group-set nonoverlapping_analysis_groups_separate_tongue_combined_forelimb \
        --groups forelimb \
        --far-side \
        --conf-threshold 0 \
        --no-json-crops \
        --auto-crop-pad 12 \
        --tag far_paw \
        --start-frame 0 \
        --n-frames 2000 || STATUS=$?
fi

if [ "${PART}" = near_paw ] || [ "${PART}" = all ]; then
    PYTHONPATH="${REPO}/src:${PYTHONPATH}" python -u -m pose2visual.qc.video \
        --eks-dir "${EKS_DIR}" \
        --config "${CONFIG}" \
        --sessions "${SESSION_IDS[@]}" \
        --group-set nonoverlapping_analysis_groups_separate_tongue_combined_forelimb \
        --groups forelimb \
        --conf-threshold 0 \
        --no-json-crops \
        --auto-crop-pad 12 \
        --no-pca \
        --tag near_paw \
        --start-frame 0 \
        --n-frames 2000 || STATUS=$?
fi

if [ "${PART}" = face ] || [ "${PART}" = all ]; then
    # near side with the json zoom boxes (the boxes exist for this group set only)
    PYTHONPATH="${REPO}/src:${PYTHONPATH}" python -u -m pose2visual.qc.video \
        --eks-dir "${EKS_DIR}" \
        --config "${CONFIG}" \
        --sessions "${SESSION_IDS[@]}" \
        --group-set nonoverlapping_analysis_groups \
        --groups eye_related mouth_related nose_pad_related ear_related \
        --start-frame 0 \
        --n-frames 2000 || STATUS=$?
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
conda deactivate
exit "${STATUS}"
