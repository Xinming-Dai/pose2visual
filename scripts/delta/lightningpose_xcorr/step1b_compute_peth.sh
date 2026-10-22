#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem 8G
#SBATCH -t 0-00:10:00
#SBATCH -J xcorr_peth
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_xcorr/step1b_compute_peth_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO="/u/xdai3/project3d/pose2visual/pose2visual"
cd "$REPO"

# The yaml is copied when the job starts, so editing it afterwards does not change this run.
# Override from the environment, e.g.:
#   sbatch --export=ALL,EID=... scripts/delta/lightningpose_xcorr/step1b_compute_peth.sh
CONFIG="${CONFIG:-$REPO/configs/encoding/litpose_encoding.yaml}"
EID="${1:-${EID:-}}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Computing PETHs, config=$CONFIG eid=${EID:-<session.eid of the config>}"

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.xcorr.peth \
    --config "$CONFIG" \
    ${EID:+--eids "$EID"}
STATUS=$?

conda deactivate
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
exit "${STATUS}"
