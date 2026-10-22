#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem 14G
#SBATCH -t 0-00:59:00
#SBATCH -J xcorr
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_xcorr/step1_run_xcorr_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO="/u/xdai3/project3d/pose2visual/pose2visual"
cd "$REPO"

# Neural-behavior cross-correlation of one session (needs step0_extract_neural_continuous.sh and
# lightningpose_encoding/step0_list_keypoints.sh for every camera). Writes one figure with every
# behavior (rows = cameras) and example curves in <video_preds_dir>/xcorr/<eid>/<YYYYmmdd_HHMMSS>/ (New York time of the run), and the
# per-neuron CSV and the arrays in <video_preds_dir>/xcorr/<eid>/. session.camera is not read.
# To redraw only the figures later, use step2_replot_xcorr.sh. If step1b_compute_peth.sh has
# written peth.<eid>.npz, the overview also gets the reward / movement-onset event strips (xcorr
# and PETH).
# The yaml is copied when the job starts, so it can be edited once the job is running (a job that
# is still queued reads it when it starts).
# Override from the environment, e.g.:
#   sbatch --export=ALL,EID=...,CAMERAS=left,EXCLUDE=,ORDER=depth scripts/delta/lightningpose_xcorr/step1_run_xcorr.sh
#   CAMERAS   cameras to analyse, space separated (default "left right"; one CPU per camera)
#   EXCLUDE   glob patterns of keypoints to leave out (default "d*" = the finger keypoints
#             d1_tip_left ... d4_tip_right); EXCLUDE= keeps everything
#   ORDER     rastermap (needs `pip install rastermap`), depth or none
#   VMAX      fixed colour scale +-VMAX (Figure 20 uses 0.6); default: from the data
CONFIG="${CONFIG:-$REPO/configs/encoding/litpose_encoding.yaml}"
EID="${1:-${EID:-}}"
VIDEO_TIMESTAMPS_DIR="${VIDEO_TIMESTAMPS_DIR:-/work/hdd/bfsr/xdai3/IBL-2view/timestamps}"
ORDER="${ORDER:-rastermap}"
# read -a splits on spaces without expanding globs such as d*
read -r -a CAMERA_LIST <<< "${CAMERAS:-left right}"
read -r -a EXCLUDE_LIST <<< "${EXCLUDE-d*}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running xcorr, config=$CONFIG eid=${EID:-<session.eid of the config>} cameras=${CAMERA_LIST[*]} exclude=${EXCLUDE_LIST[*]:-<none>} order=$ORDER"

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.xcorr.run_xcorr \
    --config "$CONFIG" \
    ${EID:+--eids "$EID"} \
    --video-timestamps-dir "$VIDEO_TIMESTAMPS_DIR" \
    --cameras "${CAMERA_LIST[@]}" \
    ${EXCLUDE_LIST[@]:+--exclude "${EXCLUDE_LIST[@]}"} \
    --order "$ORDER" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-2}" \
    ${VMAX:+--vmax "$VMAX"} \
    ${N_PERMUTATIONS:+--n-permutations "$N_PERMUTATIONS"}
STATUS=$?

conda deactivate
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
exit "${STATUS}"
