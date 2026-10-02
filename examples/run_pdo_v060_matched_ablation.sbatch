#!/bin/bash
#SBATCH --job-name=npcr-matched
#SBATCH --partition=hujun
#SBATCH --nodelist=c060
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=/share/home/lrs/codex/projects/npcrflow/logs/pdo_v060_matched_ablation-%j.out

set -euo pipefail
cd /share/home/lrs/codex/projects/npcrflow
export PYTHONPATH=/share/home/lrs/codex/projects/npcrflow/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
/share/home/lrs/.conda/envs/mybase/bin/python -u examples/run_pdo_v060_matched_ablation.py
