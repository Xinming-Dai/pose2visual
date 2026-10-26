#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem 12G
#SBATCH -t 0-00:15:00
#SBATCH -J xcorr_compute
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_xcorr/step1_compute_xcorr_%j.log
#SBATCH --export=ALL

exec 2>&1

REPO="/u/xdai3/project3d/pose2visual/pose2visual"
# sets RUN_A / RUN_B / RUN_C, EXP_SELECTED and EID from --exp and the positional eid
source "$REPO/scripts/delta/lib/parse_exp.sh"

source ~/.bashrc
conda activate beast
cd "$REPO"

# Step 1: neural-behavior cross-correlation of one session (needs step 0 and
# lightningpose_encoding/step0_list_keypoints.sh).
#   sbatch scripts/delta/lightningpose_xcorr/step1_compute_xcorr.sh [--exp A|B|C|all] <eid>
#
# Experiments (independent, can run as parallel jobs; use the same CAMERAS / EXCLUDE for all):
#   A  baseline xcorr of keypoint speed (+ PETHs)
#   B  PCA of the face group
#   C  centroid-corrected relative motion
#
# Overrides (e.g. sbatch --export=ALL,EID=...,CAMERAS=left,EXCLUDE= ...):
#   EXP                 same as --exp (the flag wins)
#   CAMERAS             cameras, space separated (default "left right")
#   EXCLUDE             glob patterns of keypoints to leave out (default "d*"); EXCLUDE= keeps all
#   ORDER               A: neuron order: rastermap (default), depth or none
#   SKIP_DEPTH_ORDER    set to 1 to skip the extra depth-order figures
#   N_PERMUTATIONS      A: permutations of the significance test (default 0 = none)
#   N_COMPONENTS        B: principal components kept (default 2)
#   SKIP_DECOMPOSITION  B: set to 1 to skip the decomposition heatmap / video
#   NO_VIDEO            B: set to 1 to skip the decomposition video
#   START_SEC           B: decomposition window start in seconds (default: auto)
#   DURATION_SEC        B: decomposition window length in seconds (default 10)
#   VIDEO_ROOT          B: folder with the <left|right>Camera.video folders
#   CROP_PAD            B: zoom the decomposition video to the keypoints plus this many pixels
#   MIN_VALID_FRACTION  B / C: leave out keypoints defined in a smaller fraction of bins (0.5)
#   NO_CENTROID_GROUPS  C: groups that keep their original speed (default "forelimb");
#                       NO_CENTROID_GROUPS= corrects every group of 2+ keypoints
#   CONFIG, VIDEO_TIMESTAMPS_DIR, EID
CONFIG="${CONFIG:-$REPO/configs/encoding/litpose_encoding.yaml}"
VIDEO_TIMESTAMPS_DIR="${VIDEO_TIMESTAMPS_DIR:-/work/hdd/bfsr/xdai3/IBL-2view/timestamps}"
ORDER="${ORDER:-rastermap}"
# read -a splits on spaces without expanding globs such as d*
read -r -a CAMERA_LIST <<< "${CAMERAS:-left right}"
read -r -a EXCLUDE_LIST <<< "${EXCLUDE-d*}"
read -r -a NO_CENTROID_LIST <<< "${NO_CENTROID_GROUPS-forelimb}"
STATUS=0

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Experiments: ${EXP_SELECTED}; config=$CONFIG eid=${EID:-<session.eid of the config>} cameras=${CAMERA_LIST[*]}"

if [ -n "$RUN_A" ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Experiment A (PETHs)"
    PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.xcorr.peth \
        --config "$CONFIG" \
        ${EID:+--eids "$EID"}
    [ $? -ne 0 ] && echo "[$(date +'%Y-%m-%d %H:%M:%S')] peth failed; the heatmap gets no PETH strips. Rerun it by hand, then step2_plot_xcorr.sh --exp A." >&2

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Experiment A (baseline xcorr), exclude=${EXCLUDE_LIST[*]:-<none>} order=$ORDER"
    PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.xcorr.run_xcorr \
        --config "$CONFIG" \
        ${EID:+--eids "$EID"} \
        --video-timestamps-dir "$VIDEO_TIMESTAMPS_DIR" \
        --cameras "${CAMERA_LIST[@]}" \
        ${EXCLUDE_LIST[@]:+--exclude "${EXCLUDE_LIST[@]}"} \
        --order "$ORDER" \
        ${SKIP_DEPTH_ORDER:+--no-depth-plot} \
        --n-jobs "${SLURM_CPUS_PER_TASK:-2}" \
        ${N_PERMUTATIONS:+--n-permutations "$N_PERMUTATIONS"}
    STATUS_A=$?
    if [ "$STATUS_A" -ne 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Experiment A (run_xcorr) failed (exit $STATUS_A); rerun it with --exp A." >&2
        [ "$STATUS" -eq 0 ] && STATUS=$STATUS_A
    fi
fi

if [ -n "$RUN_B" ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Experiment B (pca), exclude=${EXCLUDE_LIST[*]:-<none>} n-components=${N_COMPONENTS:-2}"
    PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.xcorr.experiments.pca \
        --config "$CONFIG" \
        ${EID:+--eids "$EID"} \
        --video-timestamps-dir "$VIDEO_TIMESTAMPS_DIR" \
        ${CAMERA_LIST[@]:+--cameras "${CAMERA_LIST[@]}"} \
        ${EXCLUDE_LIST[@]:+--exclude "${EXCLUDE_LIST[@]}"} \
        --n-jobs "${SLURM_CPUS_PER_TASK:-2}" \
        ${N_COMPONENTS:+--n-components "$N_COMPONENTS"} \
        ${MIN_VALID_FRACTION:+--min-valid-fraction "$MIN_VALID_FRACTION"} \
        ${SKIP_DEPTH_ORDER:+--no-depth-plot} \
        ${SKIP_DECOMPOSITION:+--no-decomposition} \
        ${NO_VIDEO:+--no-video} \
        ${START_SEC:+--window-start-sec "$START_SEC"} \
        ${DURATION_SEC:+--window-duration-sec "$DURATION_SEC"} \
        ${VIDEO_ROOT:+--video-root "$VIDEO_ROOT"} \
        ${CROP_PAD:+--crop-pad "$CROP_PAD"}
    STATUS_B=$?
    if [ "$STATUS_B" -ne 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Experiment B (pca) failed (exit $STATUS_B); rerun it with --exp B." >&2
        [ "$STATUS" -eq 0 ] && STATUS=$STATUS_B
    fi
fi

if [ -n "$RUN_C" ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Experiment C (relative motion), exclude=${EXCLUDE_LIST[*]:-<none>} no-centroid-groups=${NO_CENTROID_LIST[*]:-<none>}"
    # --no-centroid-groups is always passed: with no value it means "correct every group"
    PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.xcorr.experiments.relative_motion \
        --config "$CONFIG" \
        ${EID:+--eids "$EID"} \
        --video-timestamps-dir "$VIDEO_TIMESTAMPS_DIR" \
        ${CAMERA_LIST[@]:+--cameras "${CAMERA_LIST[@]}"} \
        ${EXCLUDE_LIST[@]:+--exclude "${EXCLUDE_LIST[@]}"} \
        --n-jobs "${SLURM_CPUS_PER_TASK:-2}" \
        --no-centroid-groups "${NO_CENTROID_LIST[@]}" \
        ${MIN_VALID_FRACTION:+--min-valid-fraction "$MIN_VALID_FRACTION"} \
        ${SKIP_DEPTH_ORDER:+--no-depth-plot}
    STATUS_C=$?
    if [ "$STATUS_C" -ne 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Experiment C (relative motion) failed (exit $STATUS_C); rerun it with --exp C." >&2
        [ "$STATUS" -eq 0 ] && STATUS=$STATUS_C
    fi
fi

conda deactivate
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
exit "${STATUS}"
