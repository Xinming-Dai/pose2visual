#!/bin/bash
#SBATCH -A bezq-delta-cpu
#SBATCH -p cpu
#SBATCH -c 3
#SBATCH --mem 20G
#SBATCH -t 0-00:20:00
#SBATCH -J litpose_list_keypoints
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_encoding/step0_list_keypoints_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

REPO="/u/xdai3/project3d/pose2visual/pose2visual"
cd "$REPO"

# Examples:
#   sbatch scripts/delta/lightningpose_encoding/step0_list_keypoints.sh   # session.eid_group of CONFIG
#   sbatch --export=ALL,EID_GROUP=sable scripts/delta/lightningpose_encoding/step0_list_keypoints.sh
#   sbatch --export=ALL,EIDS="<eid1> <eid2>" scripts/delta/lightningpose_encoding/step0_list_keypoints.sh

CONFIG="${CONFIG:-$REPO/configs/encoding/litpose_encoding.yaml}"

if [ -n "${EIDS:-}" ] && [ -n "${EID_GROUP:-}" ]; then
    echo "set only one of EIDS or EID_GROUP" >&2
    exit 1
fi
SESSION_ARGS=()
if [ -n "${EID_GROUP:-}" ]; then
    SESSION_ARGS+=(--eid-group "$EID_GROUP")
    if [ -n "${EID_GROUPS_JSON:-}" ]; then
        SESSION_ARGS+=(--eid-groups-json "$EID_GROUPS_JSON")
    fi
elif [ -n "${EIDS:-}" ]; then
    read -r -a EID_LIST <<< "$EIDS"
    SESSION_ARGS+=(--eids "${EID_LIST[@]}")
fi

N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"

OVERRIDE_ARGS=()
if [ -n "${VIDEO_PREDS_DIR:-}" ]; then
    OVERRIDE_ARGS+=(--video-preds-dir "$VIDEO_PREDS_DIR")
fi
if [ -n "${LIKELIHOOD_THRESHOLD:-}" ]; then
    OVERRIDE_ARGS+=(--likelihood-threshold "$LIKELIHOOD_THRESHOLD")
fi
if [ -n "${MAX_MASKED_FRAC:-}" ]; then
    OVERRIDE_ARGS+=(--max-masked-frac "$MAX_MASKED_FRAC")
fi
if [ -n "${FORCE_MERGE:-}" ]; then
    OVERRIDE_ARGS+=(--force-merge)
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Listing keypoints with config=$CONFIG (video_preds_dir: ${VIDEO_PREDS_DIR:-from config})"
echo "SESSION_ARGS: ${SESSION_ARGS[*]:-none, using session.eid_group of CONFIG (all csvs if unset)}"
echo "step_order (decides whether the PCA is saved, from CONFIG): $(grep -E '^step_order:' "$CONFIG" || echo 'step_order: not set (step 0 will fail)')"
echo "pca group set (from CONFIG): $(grep -E '^ +group_set:' "$CONFIG" || echo 'group_set: default')"

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -m pose2visual.encoding.list_litpose_keypoints \
    --config "$CONFIG" \
    --video-timestamps-dir "/work/hdd/bfsr/xdai3/IBL-2view/timestamps" \
    --n-jobs "$N_JOBS" \
    "${SESSION_ARGS[@]}" \
    "${OVERRIDE_ARGS[@]}"
STATUS=$?

conda deactivate

if [ "$STATUS" -ne 0 ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job failed (exit $STATUS); see the errors above" >&2
    exit "$STATUS"
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done -> <video_preds_dir>/encoding/keypoint_names/ and, with 2d_pose_pca, <video_preds_dir>/encoding/pca/<group_set>/"
