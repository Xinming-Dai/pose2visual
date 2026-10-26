#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem 14G
#SBATCH -t 0-00:10:00
#SBATCH -J xcorr_extract
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_xcorr/step0_extract_neural_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO="/u/xdai3/project3d/pose2visual/pose2visual"
cd "$REPO"

# Step 0: extract the continuous neural data and trial events of one session (run before step 1).
#   sbatch scripts/delta/lightningpose_xcorr/step0_extract_neural.sh <eid>
# Overrides: EID, CONFIG, ONE_CACHE_PATH, VIDEO_TIMESTAMPS_DIR.
CONFIG="${CONFIG:-$REPO/configs/encoding/litpose_encoding.yaml}"
EID="${1:-${EID:-}}"
ONE_CACHE_PATH="${ONE_CACHE_PATH:-/work/hdd/bfsr/xdai3/IBL_data/ONE}"
VIDEO_TIMESTAMPS_DIR="${VIDEO_TIMESTAMPS_DIR:-/work/hdd/bfsr/xdai3/IBL-2view/timestamps}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracting continuous neural data, config=$CONFIG eid=${EID:-<session.eid of the config>}"

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.xcorr.extract_neural_continuous \
    --config "$CONFIG" \
    ${EID:+--eids "$EID"} \
    --one-cache-path "$ONE_CACHE_PATH" \
    --video-timestamps-dir "$VIDEO_TIMESTAMPS_DIR" \
    --n-workers "${SLURM_CPUS_PER_TASK:-2}"
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    conda deactivate
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] extract_neural_continuous failed (exit $STATUS); trial events not attempted." >&2
    exit "$STATUS"
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracting trial events, config=$CONFIG eid=${EID:-<session.eid of the config>}"

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.xcorr.extract_trial_events \
    --config "$CONFIG" \
    ${EID:+--eids "$EID"} \
    --one-cache-path "$ONE_CACHE_PATH"
EVENTS_STATUS=$?
if [ "$EVENTS_STATUS" -ne 0 ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] extract_trial_events failed (exit $EVENTS_STATUS); no PETH strips later. Rerun it by hand." >&2
fi

conda deactivate
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
exit "${STATUS}"
