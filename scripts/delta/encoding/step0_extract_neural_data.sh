#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem 7G
#SBATCH -t 0-00:59:00
#SBATCH -J extract_neural_data
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/encoding/step0_extract_neural_data_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO="/u/xdai3/project3d/pose2visual/pose2visual"
cd "$REPO"

# Pulls spikes/behaviors for one IBL session via ONE, bins them into 1s trial windows
# synchronized to the two-view camera timestamps, filters units by firing rate, splits
# train/val/test, and writes <eid>_aligned.npz + <eid>_meta.pkl + params.json.
# Override any of these from the environment, e.g.:
#   sbatch --export=ALL,EID=...,NUM_TRIALS=... scripts/delta/encoding/sable_extract_neural_data.sh
EID="${EID:-3e6a97d3-3991-49e2-b346-6948cb4580fb}"
ONE_CACHE_PATH="${ONE_CACHE_PATH:-/work/hdd/bfsr/xdai3/IBL_data/ONE}"
VIDEO_TIMESTAMPS_DIR="${VIDEO_TIMESTAMPS_DIR:-/work/hdd/bfsr/xdai3/IBL-2view/timestamps}"
OUTPUT_DIR="${OUTPUT_DIR:-/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data}"
NUM_TRIALS="${NUM_TRIALS:-400}"
FR_THRESH="${FR_THRESH:-0.2}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracting IBL neural data for eid=$EID"

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -m pose2visual.encoding.extract_neural_data \
    --eid "$EID" \
    --one-cache-path "$ONE_CACHE_PATH" \
    --video-timestamps-dir "$VIDEO_TIMESTAMPS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-trials "$NUM_TRIALS" \
    --fr-thresh "$FR_THRESH"

conda deactivate

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done."
