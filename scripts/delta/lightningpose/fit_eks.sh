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
MODEL_DIR=/work/nvme/bfsr/xdai3/pose2visual/mighty-mouse-models/mm-anchored-lora_ibl_v14/mm-anchored-lora
SEED_DIRS=(tfall-draw0 tfall-draw1 tfall-draw2)
SESSION_IDS=(
    "51e53aff-1d5d-4182-a684-aba783d50ae5"
    "b22f694e-4a34-4142-ab9d-2556c3487086"
)

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Fitting Pose-EKS for sessions: ${SESSION_IDS[*]}"

PYTHONPATH="${REPO}/src" python -m pose2visual.lightningpose.fit_eks_latent_model \
    --model-dir "${MODEL_DIR}" \
    --seed-dirs "${SEED_DIRS[@]}" \
    --sessions "${SESSION_IDS[@]}" \
    --workers "${SLURM_CPUS_PER_TASK:-3}"
STATUS=$?

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
exit "${STATUS}"
