#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH --job-name="plot"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH -t 1:00:00
#SBATCH --mem=10G
#SBATCH --export=ALL
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/skills/pose-video/plot_comparison_%j.log

exec 2>&1
source ~/.bashrc
conda activate beast

SCRIPT=/u/xdai3/project3d/pose2visual/pose2visual/skills/pose-video/plot_comparison.py

echo "[$(date +'%Y-%m-%d %H:%M:%S')] plot_comparison"

echo "=== executing ==="
python "${SCRIPT}" \
    --model-dir /work/hdd/bfsr/xdai3/pose2visual/mighty-mouse-models/E6c-LOO-ibl-headmlp256_3seeds/face+cheese+caz+kondo+hmv+c3d+kauf_train \
    --seed-dirs vits_dinov3-seed0 vits_dinov3-seed1 vits_dinov3-seed2 \
    --session-group sable \
    --save-ensemble-median 
    # --sessions 4b00df29-3769-43be-bb40-128b1cba6d35
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done."
conda deactivate