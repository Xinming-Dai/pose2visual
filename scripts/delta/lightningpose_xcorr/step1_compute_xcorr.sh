#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem 12G
#SBATCH -t 0-04:00:00
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

# Step 1: the xcorr computations for one session (needs step0_extract_neural.sh and
# lightningpose_encoding/step0_list_keypoints.sh for every camera). B and C draw no figures --
# run step2_plot_xcorr.sh afterwards; A draws its heatmap (with PETH strips). session.camera
# is not read.
#
# --exp selects what runs (default all; a comma list such as B,C is allowed). Each experiment
# writes its own subfolder of <video_preds_dir>/xcorr/<eid>/ (step-0 inputs stay in that folder):
#   A  PETHs (pose2visual.xcorr.peth, optional, soft failure), then the baseline
#      neural-behavior cross-correlation (pose2visual.xcorr.run_xcorr), whose heatmap carries
#      the reward / movement-onset strips -> experiment_A_baseline/<time>/, plus the same
#      heatmap with neurons in depth order -> experiment_A_baseline/<time>_depth/
#   B  PCA of the face group (pose2visual.xcorr.experiments.pca --no-plot) -> experiment_B_pca/
#   C  relative motion (pose2visual.xcorr.experiments.relative_motion --no-plot)
#      -> experiment_C_relative_motion/
# A, B and C are independent: B and C recompute the original speed xcorrs they compare against
# (same code and settings as A), so they need only step 0 and can run as parallel jobs:
#   for exp in A B C; do sbatch scripts/delta/lightningpose_xcorr/step1_compute_xcorr.sh --exp $exp <eid>; done
# Use the same CAMERAS / EXCLUDE for all three so they compare like with like. A failing stage is
# logged and does not stop the others; the job exits with the status of the first failure
# (a PETH failure is only logged).
#
# The yaml is copied by each python call when it starts, so it can be edited once the job is
# running (a job that is still queued reads it when it starts).
# Override from the environment, e.g.:
#   sbatch --export=ALL,EID=...,CAMERAS=left,EXCLUDE=,ORDER=depth scripts/delta/lightningpose_xcorr/step1_compute_xcorr.sh --exp A
#   EXP                 same as --exp (the command-line flag wins)
#   CAMERAS             cameras to analyse, space separated (default "left right")
#   EXCLUDE             A / B / C: glob patterns of keypoints to leave out (default "d*" = the
#                       finger keypoints d1_tip_left ... d4_tip_right); EXCLUDE= keeps everything
#   ORDER               A: neuron order of the heatmap: rastermap (needs
#                       `pip install rastermap`), depth or none
#   SKIP_DEPTH_ORDER    A: set to 1 to skip the extra depth-order heatmap (<time>_depth/); it is
#                       never drawn twice when ORDER=depth
#   N_PERMUTATIONS      A: circular-shift permutations of the significance test (default: the
#                       yaml's xcorr.n_permutations = 0 = no test; e.g. 1000 adds p / q values)
#   N_COMPONENTS        B: number of principal components to keep for the face group (default 2)
#   MIN_VALID_FRACTION  B / C: a keypoint whose denoised/relative speed is defined in a smaller
#                       fraction of the bins is left out and reported (default 0.5)
#   NO_CENTROID_GROUPS  C: groups that keep their original speed although they have two or more
#                       keypoints (default "forelimb"); set it empty (NO_CENTROID_GROUPS=) to
#                       correct every group of 2+ keypoints
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
        --no-plot
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
        --no-plot
    STATUS_C=$?
    if [ "$STATUS_C" -ne 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Experiment C (relative motion) failed (exit $STATUS_C); rerun it with --exp C." >&2
        [ "$STATUS" -eq 0 ] && STATUS=$STATUS_C
    fi
fi

conda deactivate
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
exit "${STATUS}"
