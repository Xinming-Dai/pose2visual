#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem 14G
#SBATCH -t 0-00:59:00
#SBATCH -J xcorr_relmotion
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_xcorr/step3b_experiment_relative_motion_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO="/u/xdai3/project3d/pose2visual/pose2visual"
cd "$REPO"

# Experiment B (optional): subtract the centroid of each anatomical group with two or more
# keypoints from the keypoint coordinates, recompute the speed and cross-correlate it with the
# same neural activity as the baseline. Needs a finished step1_run_xcorr.sh run (its saved arrays,
# neural_continuous.npz and the exclude patterns it used are read) and lightningpose_encoding/
# step0_list_keypoints.sh. Writes only into <video_preds_dir>/xcorr/<eid>/
# experiment_B_relative_motion/ (the figures in a New York time subfolder); the baseline files
# are never changed. relative_groups.<eid>.md lists which groups were corrected and which kept
# their original speed.
# Override from the environment, e.g.:
#   sbatch --export=ALL,EID=...,NO_CENTROID_GROUPS="" scripts/delta/lightningpose_xcorr/step3b_experiment_relative_motion.sh
#   CAMERAS             cameras to analyse (default: all of the baseline run)
#   EXCLUDE             extra glob patterns of keypoints to leave out, on top of the baseline's
#   NO_CENTROID_GROUPS  groups that keep their original speed although they have two or more
#                       keypoints (default "forelimb": the two wrists are separate limbs).
#                       Set it empty (NO_CENTROID_GROUPS=) to correct every group of 2+ keypoints
#   MIN_VALID_FRACTION  a corrected keypoint whose relative speed is defined in a smaller fraction
#                       of the bins is left out and reported (default 0.5)
#   VMAX                shared colour scale +-VMAX of the original / relative heatmaps
#   PLOTS_SUBDIR        figure folder inside experiment_B_relative_motion (default: New York time)
CONFIG="${CONFIG:-$REPO/configs/encoding/litpose_encoding.yaml}"
EID="${1:-${EID:-}}"
VIDEO_TIMESTAMPS_DIR="${VIDEO_TIMESTAMPS_DIR:-/work/hdd/bfsr/xdai3/IBL-2view/timestamps}"
# read -a splits on spaces without expanding globs such as tongue_*
read -r -a CAMERA_LIST <<< "${CAMERAS:-}"
read -r -a EXCLUDE_LIST <<< "${EXCLUDE:-}"
read -r -a NO_CENTROID_LIST <<< "${NO_CENTROID_GROUPS-forelimb}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Experiment B (relative motion), config=$CONFIG eid=${EID:-<session.eid of the config>} cameras=${CAMERA_LIST[*]:-<all>} exclude=${EXCLUDE_LIST[*]:-<as baseline>} no-centroid-groups=${NO_CENTROID_LIST[*]:-<none>}"

# --no-centroid-groups is always passed: with no value it means "correct every group"
PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.xcorr.experiments.relative_motion \
    --config "$CONFIG" \
    ${EID:+--eids "$EID"} \
    --video-timestamps-dir "$VIDEO_TIMESTAMPS_DIR" \
    ${CAMERA_LIST[@]:+--cameras "${CAMERA_LIST[@]}"} \
    ${EXCLUDE_LIST[@]:+--exclude "${EXCLUDE_LIST[@]}"} \
    --no-centroid-groups "${NO_CENTROID_LIST[@]}" \
    ${MIN_VALID_FRACTION:+--min-valid-fraction "$MIN_VALID_FRACTION"} \
    ${VMAX:+--vmax "$VMAX"} \
    ${PLOTS_SUBDIR:+--plots-subdir "$PLOTS_SUBDIR"}
STATUS=$?

conda deactivate
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
exit "${STATUS}"
