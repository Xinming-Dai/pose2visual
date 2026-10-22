#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem 4G
#SBATCH -t 0-00:10:00
#SBATCH -J xcorr_events
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_xcorr/step0b_extract_trial_events_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO="/u/xdai3/project3d/pose2visual/pose2visual"
cd "$REPO"

# Saves the reward (feedback time of the correct trials) and movement-onset (firstMovement_times)
# times of one IBL session from its trials table, via ONE (same cache as step 0), into
# <video_preds_dir>/xcorr/<eid>/trial_events.npz. Needed once per session for the event-aligned
# PETH strips; then run step1b_compute_peth.sh and step2_replot_xcorr.sh.
# The yaml is copied when the job starts, so editing it afterwards does not change this run.
# Override from the environment, e.g.:
#   sbatch --export=ALL,EID=... scripts/delta/lightningpose_xcorr/step0b_extract_trial_events.sh
CONFIG="${CONFIG:-$REPO/configs/encoding/litpose_encoding.yaml}"
EID="${1:-${EID:-}}"
ONE_CACHE_PATH="${ONE_CACHE_PATH:-/work/hdd/bfsr/xdai3/IBL_data/ONE}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracting trial events, config=$CONFIG eid=${EID:-<session.eid of the config>}"

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.xcorr.extract_trial_events \
    --config "$CONFIG" \
    ${EID:+--eids "$EID"} \
    --one-cache-path "$ONE_CACHE_PATH"
STATUS=$?

conda deactivate
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
exit "${STATUS}"
