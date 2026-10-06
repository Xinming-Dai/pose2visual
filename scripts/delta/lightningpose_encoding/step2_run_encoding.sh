#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH -t 0-00:59:00
#SBATCH -J litpose_encoding
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_encoding/step2_run_encoding_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO="/u/xdai3/project3d/pose2visual/pose2visual"
cd "$REPO"

EID="${1:-${EID:-}}"
CONFIG="${CONFIG:-$REPO/configs/encoding/litpose_encoding.yaml}"
NEURAL_INPUT_DIR="${NEURAL_INPUT_DIR:-/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running litpose neural encoding with config=$CONFIG"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] EID=$EID"

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.encoding.run_litpose_encoding \
    --config "$CONFIG" \
    --neural_input_dir "$NEURAL_INPUT_DIR" \
    ${EID:+--eid "$EID"} \
    ${RERUN_BASE:+--rerun_base}

conda deactivate

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done."
