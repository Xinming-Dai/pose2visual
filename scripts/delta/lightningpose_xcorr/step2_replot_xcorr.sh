#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem 3G
#SBATCH -t 0-00:10:00
#SBATCH -J xcorr_replot
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_xcorr/step2_replot_xcorr_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO="/u/xdai3/project3d/pose2visual/pose2visual"
cd "$REPO"

# Redraws the figures of a finished run (step1_run_xcorr.sh) from the saved arrays and csv, into
# <video_preds_dir>/xcorr/<eid>/<PLOTS_SUBDIR>/, by default a new folder named by the New York time
# of the job (YYYYmmdd_HHMMSS). Nothing is recomputed and nothing but that folder is written, so it
# takes seconds. The neuron order stored by the run is reused.
# Override from the environment, e.g.:
#   sbatch --export=ALL,EID=...,VMAX=0.6,EXCLUDE="tongue_*",PLOTS_SUBDIR=plots_vmax06 scripts/delta/lightningpose_xcorr/step2_replot_xcorr.sh
#   CAMERAS       cameras (rows) to draw, space separated (default: all of the run)
#   EXCLUDE       glob patterns of behaviors to hide, space separated (default: none)
#   ORDER         saved (default), depth or none
#   VMAX          fixed colour scale +-VMAX (Figure 20 uses 0.6); default: from the data
#   PLOTS_SUBDIR  figure folder inside the session folder (default: New York time, YYYYmmdd_HHMMSS)
#   N_EXAMPLES    number of example curves (default: 0 = none)
#   NO_PETH       set to 1 to leave out the reward / movement-onset event strips, xcorr and PETH
#                 (they are drawn when step1b_compute_peth.sh has written peth.<eid>.npz)
#   PETH_VMAX     PETH colour scale +-PETH_VMAX (default 1 = a doubling of the rate)
CONFIG="${CONFIG:-$REPO/configs/encoding/litpose_encoding.yaml}"
EID="${1:-${EID:-}}"
ORDER="${ORDER:-saved}"
PETH_VMAX="${PETH_VMAX:-2}"
# read -a splits on spaces without expanding globs such as tongue_*
read -r -a CAMERA_LIST <<< "${CAMERAS:-}"
read -r -a EXCLUDE_LIST <<< "${EXCLUDE:-}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Replotting xcorr, config=$CONFIG eid=${EID:-<session.eid of the config>} cameras=${CAMERA_LIST[*]:-<all>} exclude=${EXCLUDE_LIST[*]:-<none>} order=$ORDER"

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.xcorr.replot \
    --config "$CONFIG" \
    ${EID:+--eids "$EID"} \
    ${CAMERA_LIST[@]:+--cameras "${CAMERA_LIST[@]}"} \
    ${EXCLUDE_LIST[@]:+--exclude "${EXCLUDE_LIST[@]}"} \
    --order "$ORDER" \
    ${VMAX:+--vmax "$VMAX"} \
    ${PLOTS_SUBDIR:+--plots-subdir "$PLOTS_SUBDIR"} \
    ${N_EXAMPLES:+--n-examples "$N_EXAMPLES"} \
    ${NO_PETH:+--no-peth} \
    ${PETH_VMAX:+--peth-vmax "$PETH_VMAX"}
STATUS=$?

conda deactivate
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
exit "${STATUS}"
