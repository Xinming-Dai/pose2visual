#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 3
#SBATCH --mem 8G
#SBATCH -t 0-00:20:00
#SBATCH -J xcorr_pca_recon
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_xcorr/step3_render_pca_reconstruction_%j.log
#SBATCH --export=ALL

exec 2>&1

REPO="/u/xdai3/project3d/pose2visual/pose2visual"
source ~/.bashrc
conda activate beast
cd "$REPO"

# Step 3: PCA decomposition of the face of one session (Experiment B: heatmap, video, md),
# recomputed on its own, e.g. for another window, crop or rank. Needs only
# lightningpose_encoding/step0_list_keypoints.sh and the camera timestamps.
#   sbatch scripts/delta/lightningpose_xcorr/step3_render_pca_reconstruction.sh <eid>
#   sbatch --export=ALL,CAMERAS=left,START_SEC=600,DURATION_SEC=8 scripts/delta/lightningpose_xcorr/step3_render_pca_reconstruction.sh <eid>
#
# Overrides:
#   CAMERAS       cameras, space separated (default "left right"; the left one picks the window)
#   EXCLUDE       glob patterns of keypoints to leave out (default "d*"); EXCLUDE= keeps all
#   N_COMPONENTS  rank of the reconstruction (default 2)
#   START_SEC     window start in seconds (default: auto-picked on the left camera)
#   DURATION_SEC  window length in seconds (default 10)
#   MARGIN_SEC    seconds skipped at both ends when auto-picking (default 30)
#   VMAX          heatmap colour scale +-VMAX pixels (default: from the data)
#   CROP_PAD      zoom the video to the keypoints plus this many pixels (default: whole frame)
#   VIDEO_ROOT    folder with the <left|right>Camera.video folders
#   PLOTS_SUBDIR  run name (default: New York time)
#   NO_VIDEO / NO_HEATMAP   set to 1 to skip the video / the heatmap
#   CONFIG, VIDEO_TIMESTAMPS_DIR, EID
CONFIG="${CONFIG:-$REPO/configs/encoding/litpose_encoding.yaml}"
EID="${1:-${EID:-}}"
VIDEO_TIMESTAMPS_DIR="${VIDEO_TIMESTAMPS_DIR:-/work/hdd/bfsr/xdai3/IBL-2view/timestamps}"
# read -a splits on spaces without expanding globs such as d*
read -r -a CAMERA_LIST <<< "${CAMERAS:-left right}"
read -r -a EXCLUDE_LIST <<< "${EXCLUDE-d*}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] PCA decomposition, config=$CONFIG eid=${EID:-<session.eid of the config>} cameras=${CAMERA_LIST[*]} exclude=${EXCLUDE_LIST[*]:-<none>}"

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.xcorr.experiments.pca_reconstruction \
    --config "$CONFIG" \
    ${EID:+--eids "$EID"} \
    --video-timestamps-dir "$VIDEO_TIMESTAMPS_DIR" \
    --cameras "${CAMERA_LIST[@]}" \
    ${EXCLUDE_LIST[@]:+--exclude "${EXCLUDE_LIST[@]}"} \
    ${N_COMPONENTS:+--n-components "$N_COMPONENTS"} \
    ${START_SEC:+--window-start-sec "$START_SEC"} \
    ${DURATION_SEC:+--window-duration-sec "$DURATION_SEC"} \
    ${MARGIN_SEC:+--window-margin-sec "$MARGIN_SEC"} \
    ${VMAX:+--window-vmax "$VMAX"} \
    ${CROP_PAD:+--crop-pad "$CROP_PAD"} \
    ${VIDEO_ROOT:+--video-root "$VIDEO_ROOT"} \
    ${PLOTS_SUBDIR:+--plots-subdir "$PLOTS_SUBDIR"} \
    ${NO_VIDEO:+--no-video} \
    ${NO_HEATMAP:+--no-decomposition-heatmap}
STATUS=$?

conda deactivate
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
exit "${STATUS}"
