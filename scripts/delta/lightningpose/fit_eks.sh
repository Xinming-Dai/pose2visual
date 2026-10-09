#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH --job-name="fit_eks"
#SBATCH --partition=cpu
#SBATCH -c 3
#SBATCH -t 1:59:00
#SBATCH --mem=10G
#SBATCH --export=ALL
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose/fit_eks_%j.log

exec 2>&1
source ~/.bashrc
conda activate beast

REPO=/u/xdai3/project3d/pose2visual/pose2visual
MODEL_DIR=/work/hdd/bfsr/xdai3/pose2visual/mighty-mouse-models/E6c-LOO-ibl-headmlp256_3seeds/face+cheese+caz+kondo+hmv+c3d+kauf_train
SEED_DIRS=(vits_dinov3-seed0 vits_dinov3-seed1 vits_dinov3-seed2)
SESSION_IDS=(
    "9b528ad0-4599-4a55-9148-96cc1d93fb24"
)

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Fitting Pose-EKS for sessions: ${SESSION_IDS[*]}"

for SESSION_ID in "${SESSION_IDS[@]}"; do
    echo "=== ${SESSION_ID} ==="
    PYTHONPATH="${REPO}/src" python -m pose2visual.lightningpose.fit_eks_latent_model \
        --model-dir "${MODEL_DIR}" \
        --seed-dirs "${SEED_DIRS[@]}" \
        --session "${SESSION_ID}" \
        --workers "${SLURM_CPUS_PER_TASK:-3}"
done

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done."
