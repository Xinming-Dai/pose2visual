#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem 3G
#SBATCH -t 0-00:25:00
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

# Step 2: redraws the figures of a finished step1_compute_xcorr.sh run -- Experiment A
# (baseline), B (pca) and C (relative motion) -- from their saved arrays/csv/json. Nothing is
# recomputed, so this is fast. --exp A|B|C|all (default all; a comma list such as B,C is allowed)
# selects which ones are redrawn, e.g. sbatch step2_plot_xcorr.sh --exp B <eid>. Each plot type is drawn once in $ORDER (default: saved, i.e. the
# compute step's own order, usually rastermap) and, unless SKIP_DEPTH_ORDER is set, a second time
# with neurons in depth order (ascending position along the probe). Each figure folder is named
# "<PLOTS_SUBDIR>_<order>" (e.g. 20261021_153012_rastermap and 20261021_153012_depth), so the
# figures of one plotting run are easy to find together and the order is readable from the name:
# <video_preds_dir>/xcorr/<eid>/experiment_A_baseline/<PLOTS_SUBDIR>_<order>/,
# .../experiment_B_pca/<PLOTS_SUBDIR>_<order>/ and
# .../experiment_C_relative_motion/<PLOTS_SUBDIR>_<order>/.
# In the saved order, B and C are drawn in A's neuron order when A has run on the same neurons
# (else in depth order). Each of the three plot types, in each order, is independent: a missing or failed one is logged
# and does not stop the others, and does not fail the job.
# Override from the environment, e.g.:
#   sbatch --export=ALL,EID=...,VMAX=0.6,PLOTS_SUBDIR=plots_vmax06 scripts/delta/lightningpose_xcorr/step2_plot_xcorr.sh --exp A
#   EXP              same as --exp (the command-line flag wins)
#   CAMERAS          cameras (rows) to draw, space separated (default: all of the run)
#   EXCLUDE          glob patterns of baseline behaviors to hide, space separated (default: none;
#                    Experiments B / C do not support --exclude at replot time)
#   ORDER            baseline neuron order of the first pass: saved (default), depth or none
#   SKIP_DEPTH_ORDER set to 1 to skip the extra depth-order pass (e.g. when ORDER=depth already)
#   VMAX             fixed colour scale +-VMAX shared by all three (Figure 20 uses 0.6); default:
#                    each run's own rule, from its data
#   PLOTS_SUBDIR     run name of the figure folders inside each session/experiment folder
#                    (default: New York time); the neuron order is appended ("_rastermap",
#                    "_depth", "_none")
#   N_EXAMPLES       baseline example curves (default: 0 = none)
#   NO_PETH          set to 1 to leave the baseline's reward / movement-onset event strips out
#   PETH_VMAX        baseline PETH colour scale +-PETH_VMAX (default 2 = a tripling of the rate)
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
