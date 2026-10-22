#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH --job-name="pca_eks"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH -t 0:30:00
#SBATCH --mem=12G
#SBATCH --export=ALL
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose/pca_eks_%j.log

exec 2>&1
source ~/.bashrc
conda activate beast

REPO=/u/xdai3/project3d/pose2visual/pose2visual
cd "$REPO"

# Per-group PCA reprojection of the Pose-EKS keypoints (run fit_eks.sh first). Settings (likelihood
# threshold, max masked fraction, position.columns, body_parts.json) come from CONFIG; the number of
# components of each group is PCA_n_components of GROUP_SET in body_parts.json.
# Output: <run>/pca/<GROUP_SET>/pca_positions.<eid>.<camera>.npz and pca_config.<eid>.<camera>.json
# (how it was processed). qc/render_video.sh draws it as a thin red trace when --group-set matches.
CONFIG="${CONFIG:-configs/encoding/litpose_encoding.yaml}"
GROUP_SET="${GROUP_SET:-nonoverlapping_analysis_groups}"
EKS_DIR=/work/hdd/bfsr/xdai3/pose2visual/mighty-mouse-models/E6c-LOO-ibl-headmlp256_3seeds/face+cheese+caz+kondo+hmv+c3d+kauf_train/eks/20261013_173251/eks
SESSION_IDS=(
    "9b528ad0-4599-4a55-9148-96cc1d93fb24" "51e53aff-1d5d-4182-a684-aba783d50ae5" "b22f694e-4a34-4142-ab9d-2556c3487086"
)

echo "[$(date +'%Y-%m-%d %H:%M:%S')] PCA of Pose-EKS keypoints: group set ${GROUP_SET}, sessions: ${SESSION_IDS[*]}"

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.lightningpose.pca_eks \
    --eks-dir "${EKS_DIR}" \
    --config "${CONFIG}" \
    --group-set "${GROUP_SET}" \
    --sessions "${SESSION_IDS[@]}"
STATUS=$?

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
conda deactivate
exit "${STATUS}"
