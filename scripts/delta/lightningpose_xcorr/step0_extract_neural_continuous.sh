#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem 14G
#SBATCH -t 0-00:20:00
#SBATCH -J xcorr_neural
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_xcorr/step0_extract_neural_continuous_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO="/u/xdai3/project3d/pose2visual/pose2visual"
cd "$REPO"

# Bins the spikes of one IBL session at 60 Hz over the WHOLE video (every contiguous 1 s interval,
# not the 400 sampled trials of encoding/step0_extract_neural_data.sh), with the same camera
# synchronization check and bin edges, and writes
# <video_preds_dir>/xcorr/<eid>/neural_continuous.npz (+ meta pkl, params json). The spikes do not
# depend on the camera, so nothing here reads session.camera. The yaml is copied when the job
# starts, so editing it afterwards does not change this run.
# Override from the environment, e.g.:
#   sbatch --export=ALL,EID=... scripts/delta/lightningpose_xcorr/step0_extract_neural_continuous.sh
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

conda deactivate
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
exit "${STATUS}"
