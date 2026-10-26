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
MODEL_DIR=/work/nvme/bfsr/xdai3/pose2visual/mighty-mouse-models/E6c-LOO-ibl-headmlp256_3seeds/face+cheese+caz+kondo+hmv+c3d+kauf_train
SEED_DIRS=(vits_dinov3-seed0 vits_dinov3-seed1 vits_dinov3-seed2)
SESSION_IDS=(
    "3e6a97d3-3991-49e2-b346-6948cb4580fb"
    "4b00df29-3769-43be-bb40-128b1cba6d35"
    "5dcee0eb-b34d-4652-acc3-d10afc6eae68"
    "72cb5550-43b4-4ef0-add5-e4adfdfb5e02"
    "781b35fd-e1f0-4d14-b2bb-95b7263082bb"
    "ecb5520d-1358-434c-95ec-93687ecd1396"
    "f312aaec-3b6f-44b3-86b4-3a0c119c0438"

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
