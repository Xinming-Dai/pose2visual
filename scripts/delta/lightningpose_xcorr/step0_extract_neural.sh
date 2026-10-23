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

# Step 0: everything needed before the xcorr compute step, for one IBL session.
#
# 1. Bins the spikes at 60 Hz over the WHOLE video (every contiguous 1 s interval, not the 400
#    sampled trials of encoding/step0_extract_neural_data.sh), with the same camera
#    synchronization check and bin edges, and writes
#    <video_preds_dir>/xcorr/<eid>/neural_continuous.npz (+ meta pkl, params json). Required by
#    everything downstream (step1_compute_xcorr.sh); a failure here fails the job.
# 2. Saves the reward (feedback time of the correct trials) and movement-onset
#    (firstMovement_times) times from the session's trials table, via ONE (same cache as 1.),
#    into <video_preds_dir>/xcorr/<eid>/trial_events.npz. Only needed for the event-aligned PETH
#    strips (step1_compute_xcorr.sh's peth sub-step); a failure here is logged and does not fail
#    the job.
#
# The yaml is copied by each python call when it starts, so editing it afterwards does not change
# this run.
# Override from the environment, e.g.:
#   sbatch --export=ALL,EID=... scripts/delta/lightningpose_xcorr/step0_extract_neural.sh
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
