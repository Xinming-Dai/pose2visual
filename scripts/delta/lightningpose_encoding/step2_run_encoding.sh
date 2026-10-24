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

# freeze the config once
CONFIG_SOURCE="$CONFIG"
CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "$CONFIG"' EXIT
cp "$CONFIG_SOURCE" "$CONFIG"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running litpose neural encoding with config=$CONFIG_SOURCE (frozen copy $CONFIG)"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] EID=$EID"

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.encoding.run_litpose_encoding \
    --config "$CONFIG" \
    --neural_input_dir "$NEURAL_INPUT_DIR" \
    ${EID:+--eid "$EID"} \
    ${RERUN_BASE:+--rerun_base}
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    conda deactivate
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job failed (exit $STATUS); see the errors above, no plots made" >&2
    exit "$STATUS"
fi

# the plots compare base with base + additional, so a base-only run (empty additional_keypoints)
# has nothing to plot
N_ADDITIONAL="$(PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -c "
import sys
from pose2visual.encoding.litpose_keypoints import load_encoding_yaml
print(len(load_encoding_yaml(sys.argv[1]).get('additional_keypoints') or []))
" "$CONFIG" 2>/dev/null)"
if [ "${N_ADDITIONAL:-1}" -eq 0 ] && [ -z "${SKIP_PLOT:-}" ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] No additional_keypoints: base-only run, skipping the base vs additional plots"
    SKIP_PLOT=1
fi

# SKIP_PLOT=1 turns all plots off. A failed plot is logged but does not fail the finished encoding.
if [ -z "${SKIP_PLOT:-}" ]; then
    PLOT_DIR="$REPO/scripts/delta/analysis/encoding_plotting"

    # per-session plots only need this session, so they are made right away
    for PLOT_SCRIPT in plot_psth_activity_raster.py plot_bps_scatter.py; do
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Plotting $PLOT_SCRIPT for the session"
        PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u "$PLOT_DIR/$PLOT_SCRIPT" \
            --config "$CONFIG" \
            ${EID:+--eid "$EID"} \
            --no-show
        PLOT_STATUS=$?
        if [ "$PLOT_STATUS" -ne 0 ]; then
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Encoding finished but $PLOT_SCRIPT failed (exit $PLOT_STATUS); rerun it by hand" >&2
        fi
    done

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Checking whether all sessions of the group are done (bar plot)"
    PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u \
        "$PLOT_DIR/plot_encoding_base_vs_additional.py" \
        --config "$CONFIG" \
        --only-if-complete \
        --no-show
    PLOT_STATUS=$?
    if [ "$PLOT_STATUS" -ne 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Encoding finished but the bar plot failed (exit $PLOT_STATUS); rerun plot_encoding_base_vs_additional.py by hand" >&2
    fi
fi

conda deactivate

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done."
