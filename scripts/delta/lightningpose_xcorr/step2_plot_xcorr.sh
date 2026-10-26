#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem 3G
#SBATCH -t 0-00:10:00
#SBATCH -J xcorr_plot
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_xcorr/step2_plot_xcorr_%j.log
#SBATCH --export=ALL

exec 2>&1

REPO="/u/xdai3/project3d/pose2visual/pose2visual"
# sets RUN_A / RUN_B / RUN_C, EXP_SELECTED and EID from --exp and the positional eid
source "$REPO/scripts/delta/lib/parse_exp.sh"

source ~/.bashrc
conda activate beast
cd "$REPO"

# Step 2: redraw the figures of a finished step 1 run from its saved results (no recomputation).
#   sbatch scripts/delta/lightningpose_xcorr/step2_plot_xcorr.sh [--exp A|B|C|all] <eid>
# Draws once in $ORDER and, unless SKIP_DEPTH_ORDER is set, once in depth order. B also redraws
# heatmap.pca_decomposition and its decomposition.<eid>.md; the video comes from step3 only.
#
# Overrides (e.g. sbatch --export=ALL,EID=...,VMAX=0.6,PLOTS_SUBDIR=vmax06 ...):
#   EXP              same as --exp (the flag wins)
#   CAMERAS          cameras to draw, space separated (default: all of the run)
#   EXCLUDE          A: glob patterns of keypoints to hide
#   ORDER            neuron order: saved (default, A's order), depth or none
#   SKIP_DEPTH_ORDER set to 1 to skip the depth-order pass
#   VMAX             fixed colour scale +-VMAX (default: from the data)
#   PLOTS_SUBDIR     run name of the figure folders (default: New York time)
#   N_EXAMPLES       A: number of example curves (default 0)
#   NO_PETH          A: set to 1 to leave out the reward / movement-onset strips
#   PETH_VMAX        A: PETH colour scale +-PETH_VMAX (default 2)
#   CONFIG, EID
CONFIG="${CONFIG:-$REPO/configs/encoding/litpose_encoding.yaml}"
ORDER="${ORDER:-saved}"
PETH_VMAX="${PETH_VMAX:-2}"
# one folder name for the whole job, so every selected experiment carries the same run time
PLOTS_SUBDIR="${PLOTS_SUBDIR:-$(TZ=America/New_York date +%Y%m%d_%H%M%S)}"
# read -a splits on spaces without expanding globs such as tongue_*
read -r -a CAMERA_LIST <<< "${CAMERAS:-}"
read -r -a EXCLUDE_LIST <<< "${EXCLUDE:-}"

# redraws the selected experiments in one order, into one plots subdir
replot_one_order() {
    local order="$1" subdir="$2"

    if [ -n "$RUN_A" ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Replotting Experiment A (baseline xcorr), config=$CONFIG eid=${EID:-<session.eid of the config>} cameras=${CAMERA_LIST[*]:-<all>} exclude=${EXCLUDE_LIST[*]:-<none>} order=$order plots_subdir=$subdir"
        PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.xcorr.replot \
            --config "$CONFIG" \
            ${EID:+--eids "$EID"} \
            ${CAMERA_LIST[@]:+--cameras "${CAMERA_LIST[@]}"} \
            ${EXCLUDE_LIST[@]:+--exclude "${EXCLUDE_LIST[@]}"} \
            --order "$order" \
            ${VMAX:+--vmax "$VMAX"} \
            --plots-subdir "$subdir" \
            ${N_EXAMPLES:+--n-examples "$N_EXAMPLES"} \
            ${NO_PETH:+--no-peth} \
            --peth-vmax "$PETH_VMAX"
        [ $? -ne 0 ] && echo "[$(date +'%Y-%m-%d %H:%M:%S')] Experiment A (baseline) replot failed (order=$order); rerun it by hand." >&2
    fi

    if [ -n "$RUN_B" ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Replotting Experiment B (pca), order=$order"
        PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.xcorr.experiments.replot_pca \
            --config "$CONFIG" \
            ${EID:+--eids "$EID"} \
            ${CAMERA_LIST[@]:+--cameras "${CAMERA_LIST[@]}"} \
            --order "$order" \
            ${VMAX:+--vmax "$VMAX"} \
            --plots-subdir "$subdir"
        [ $? -ne 0 ] && echo "[$(date +'%Y-%m-%d %H:%M:%S')] Experiment B (pca) replot failed (order=$order; no saved run? rerun step1 --exp B first); skipping." >&2
    fi

    if [ -n "$RUN_C" ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Replotting Experiment C (relative motion), order=$order"
        PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.xcorr.experiments.replot_relative_motion \
            --config "$CONFIG" \
            ${EID:+--eids "$EID"} \
            ${CAMERA_LIST[@]:+--cameras "${CAMERA_LIST[@]}"} \
            --order "$order" \
            ${VMAX:+--vmax "$VMAX"} \
            --plots-subdir "$subdir"
        [ $? -ne 0 ] && echo "[$(date +'%Y-%m-%d %H:%M:%S')] Experiment C (relative motion) replot failed (order=$order; no saved run? rerun step1 --exp C first); skipping." >&2
    fi
}

replot_one_order "$ORDER" "$PLOTS_SUBDIR"

if [ -n "${SKIP_DEPTH_ORDER:-}" ] || [ "$ORDER" = "depth" ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Depth-order pass skipped (SKIP_DEPTH_ORDER set or ORDER already depth)."
else
    replot_one_order "depth" "$PLOTS_SUBDIR"
fi

conda deactivate
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done."
