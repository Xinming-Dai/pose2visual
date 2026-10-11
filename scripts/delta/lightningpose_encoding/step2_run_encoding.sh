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

# Usage: sbatch scripts/delta/lightningpose_encoding/step2_run_encoding.sh <eid>  
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
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    conda deactivate
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job failed (exit $STATUS); see the errors above, no plot made" >&2
    exit "$STATUS"
fi

# the sessions are submitted one by one: the job that finds every session of the config's
# session.eid_group finished makes the plot, the others log which sessions are still missing.
# SKIP_PLOT=1 turns this off. A failed plot is logged but does not fail the finished encoding.
if [ -z "${SKIP_PLOT:-}" ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Checking whether all sessions of the group are done (plot)"
    PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u \
        "$REPO/scripts/delta/analysis/encoding_plotting/plot_encoding_base_vs_additional.py" \
        --config "$CONFIG" \
        --only-if-complete \
        --no-show
    PLOT_STATUS=$?
    if [ "$PLOT_STATUS" -ne 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Encoding finished but the plot failed (exit $PLOT_STATUS); rerun plot_encoding_base_vs_additional.py by hand" >&2
    fi
fi

conda deactivate

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done."
