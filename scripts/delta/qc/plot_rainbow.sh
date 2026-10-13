#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH --job-name="plot"
#SBATCH --partition=cpu
#SBATCH -c 4
#SBATCH -t 00:29:00
#SBATCH --mem=24G
#SBATCH --export=ALL
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/qc/plot_rainbow_%j.log

exec 2>&1
source ~/.bashrc
conda activate beast

REPO=/u/xdai3/project3d/pose2visual/pose2visual
cd "$REPO"

CONFIG="${CONFIG:-configs/encoding/litpose_encoding.yaml}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] plot_rainbow"

echo "=== executing ==="
PYTHONPATH="${REPO}/src:${PYTHONPATH}" python -u -m pose2visual.qc.rainbow \
    --model-dir /work/hdd/bfsr/xdai3/pose2visual/mighty-mouse-models/ibl-ears \
    --seed-dirs 2026-10-10_19-41-30_seed-0 2026-10-10_20-02-37_seed-1 2026-10-10_20-23-51_seed-2 \
    --config "${CONFIG}" \
    --session-group ear_left \
    --group-set nonoverlapping_analysis_groups_ibl_ears \
    --save-ensemble-median
    # --sessions 4b00df29-3769-43be-bb40-128b1cba6d35
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done."
conda deactivate
