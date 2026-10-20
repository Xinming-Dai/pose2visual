#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH --job-name="qc_showcase"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH -t 00:10:00
#SBATCH --mem=8G
#SBATCH --export=ALL
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/qc/render_showcase_%j.log

exec 2>&1
source ~/.bashrc
conda activate beast

REPO=/u/xdai3/project3d/pose2visual/pose2visual
cd "$REPO"

EKS_DIR=/work/hdd/bfsr/xdai3/pose2visual/mighty-mouse-models/mm_ibl_ear/20261013_173251_n_20261014_141544
SESSION_IDS=(
    "51e53aff-1d5d-4182-a684-aba783d50ae5"
    # "9b528ad0-4599-4a55-9148-96cc1d93fb24"
    # "b22f694e-4a34-4142-ab9d-2556c3487086"
)

echo "[$(date +'%Y-%m-%d %H:%M:%S')] qc showcase for sessions: ${SESSION_IDS[*]}"

echo "=== executing ==="
# per session in <EKS_DIR>/qc/showcase/ (png + pdf):
#   kp_frame.<session>.frame000200           left | right frame, keypoints above --conf-threshold
#   kp_traces.<session>.frame000200.pm2p5s   same frames + x/y traces, 2.5 s before to 2.5 s after
# --anchor-frame N      frame to show (default: 200)
# --half-window-sec S   seconds of trace on each side (default: 2.5)
# --figures frames      only the labeled frames (or: traces)
PYTHONPATH="${REPO}/src:${PYTHONPATH}" python -u -m pose2visual.qc.showcase \
    --eks-dir "${EKS_DIR}" \
    --sessions "${SESSION_IDS[@]}" \
    --anchor-frame 200 \
    --half-window-sec 2.5
STATUS=$?

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
conda deactivate
exit "${STATUS}"
