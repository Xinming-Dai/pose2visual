#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem 3G
#SBATCH -t 0-00:10:00
#SBATCH -J xcorr_rank1
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_xcorr/step3a_experiment_rank1_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO="/u/xdai3/project3d/pose2visual/pose2visual"
cd "$REPO"

# Experiment A (optional): subtract the rank-1 approximation from the baseline xcorrs of each
# camera. Needs a finished step1_run_xcorr.sh run; reads its xcorr_arrays.<eid>.npz and writes
# only into <video_preds_dir>/xcorr/<eid>/experiment_A_rank1/ (the figures in a New York time
# subfolder). The baseline files are never changed.
# Override from the environment, e.g.:
#   sbatch --export=ALL,EID=...,CAMERAS=left,EXCLUDE="tongue_*" scripts/delta/lightningpose_xcorr/step3a_experiment_rank1.sh
#   CAMERAS       cameras to decompose, each on its own (default: all of the baseline run)
#   EXCLUDE       glob patterns of behaviors to leave out (default: none)
#   VMAX          shared colour scale +-VMAX of original / rank-1 / residual (default: from the data)
#   PLOTS_SUBDIR  figure folder inside experiment_A_rank1 (default: New York time, YYYYmmdd_HHMMSS)
#   N_EXAMPLES    number of residual example curves (default: 0 = none)
CONFIG="${CONFIG:-$REPO/configs/encoding/litpose_encoding.yaml}"
EID="${1:-${EID:-}}"
# read -a splits on spaces without expanding globs such as tongue_*
read -r -a CAMERA_LIST <<< "${CAMERAS:-}"
read -r -a EXCLUDE_LIST <<< "${EXCLUDE:-}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Experiment A (rank 1), config=$CONFIG eid=${EID:-<session.eid of the config>} cameras=${CAMERA_LIST[*]:-<all>} exclude=${EXCLUDE_LIST[*]:-<none>}"

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.xcorr.experiments.rank1 \
    --config "$CONFIG" \
    ${EID:+--eids "$EID"} \
    ${CAMERA_LIST[@]:+--cameras "${CAMERA_LIST[@]}"} \
    ${EXCLUDE_LIST[@]:+--exclude "${EXCLUDE_LIST[@]}"} \
    ${VMAX:+--vmax "$VMAX"} \
    ${PLOTS_SUBDIR:+--plots-subdir "$PLOTS_SUBDIR"} \
    ${N_EXAMPLES:+--n-examples "$N_EXAMPLES"}
STATUS=$?

conda deactivate
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
exit "${STATUS}"
