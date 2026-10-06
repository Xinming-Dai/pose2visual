#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem 10G
#SBATCH -t 0-00:59:00
#SBATCH -J litpose_dynamic_vars
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_encoding/step1_extract_dynamic_vars_trials_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO="/u/xdai3/project3d/pose2visual/pose2visual"
cd "$REPO"

# For every session in EIDS and every camera in CAMERAS (the loop runs inside python), builds
# $VIDEO_PREDS_DIR/<camera>_$OUTPUT_NAME_SUFFIX/<eid>/dynamic_vars.npz: the speed of every usable
# keypoint of that camera (from your Lightning Pose checkpoint), binned onto the intervals of
# <eid>_aligned.npz. The two cameras get separate folders (left_encoding, right_encoding), and the
# yaml `output_name` of step 2 must equal the folder name. A failing session/camera does not stop
# the others; failures are listed at the end and make the job exit non-zero. Run
# step0_list_keypoints.sh first. Override from the environment (space-separated lists), e.g.:
#   sbatch --export=ALL,EIDS="<eid1> <eid2>",CAMERAS=left scripts/delta/lightningpose_encoding/step1_extract_dynamic_vars_trials.sh
VIDEO_PREDS_DIR="${VIDEO_PREDS_DIR:-/work/hdd/bfsr/xdai3/pose2visual/mighty-mouse-models/E3a-headmlp256/face+ibl+cheese+caz+kondo+hmv+c3d+kauf_train/supervised/sampling-T2/tf1/vits_dinov3/seed0/video_preds}"
NEURAL_INPUT_DIR="${NEURAL_INPUT_DIR:-/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data}"
VIDEO_TIMESTAMPS_DIR="${VIDEO_TIMESTAMPS_DIR:-/work/hdd/bfsr/xdai3/IBL-2view/timestamps}"
OUTPUT_NAME_SUFFIX="${OUTPUT_NAME_SUFFIX:-encoding}"

DEFAULT_EID_LIST=(
    3e6a97d3-3991-49e2-b346-6948cb4580fb
    4b00df29-3769-43be-bb40-128b1cba6d35
    5dcee0eb-b34d-4652-acc3-d10afc6eae68
    72cb5550-43b4-4ef0-add5-e4adfdfb5e02
    781b35fd-e1f0-4d14-b2bb-95b7263082bb
    ecb5520d-1358-434c-95ec-93687ecd1396
    f312aaec-3b6f-44b3-86b4-3a0c119c0438
)
# EIDS / CAMERAS from the environment are single-line, space-separated strings
if [ -n "${EIDS:-}" ]; then
    read -r -a EID_LIST <<< "$EIDS"
else
    EID_LIST=("${DEFAULT_EID_LIST[@]}")
fi
read -r -a CAMERA_LIST <<< "${CAMERAS:-left right}"

if [ "${#EID_LIST[@]}" -eq 0 ] || [ "${#CAMERA_LIST[@]}" -eq 0 ]; then
    echo "EID_LIST or CAMERA_LIST is empty; check EIDS / CAMERAS" >&2
    exit 1
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracting litpose keypoint dynamic vars"
echo "VIDEO_PREDS_DIR=$VIDEO_PREDS_DIR"
echo "EIDS (${#EID_LIST[@]}): ${EID_LIST[*]}"
echo "CAMERAS: ${CAMERA_LIST[*]}"

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.encoding.extract_litpose_dynamic_vars_trials \
    --eids "${EID_LIST[@]}" \
    --cameras "${CAMERA_LIST[@]}" \
    --video-preds-dir "$VIDEO_PREDS_DIR" \
    --output-name "{camera}_${OUTPUT_NAME_SUFFIX}" \
    --neural-input-dir "$NEURAL_INPUT_DIR" \
    --video-timestamps-dir "$VIDEO_TIMESTAMPS_DIR"
STATUS=$?

conda deactivate

if [ "$STATUS" -ne 0 ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job failed (exit $STATUS); see the errors above" >&2
    exit "$STATUS"
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done -> $VIDEO_PREDS_DIR/<camera>_$OUTPUT_NAME_SUFFIX/<eid>/dynamic_vars.npz"
