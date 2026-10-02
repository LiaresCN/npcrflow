#!/bin/bash
#SBATCH --job-name=npcr-v060-ann
#SBATCH --partition=hujun
#SBATCH --nodelist=c060
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/share/home/lrs/codex/projects/npcrflow/logs/pdo_v060_resolution_ablation-%j.out

set -euo pipefail
cd /share/home/lrs/codex/projects/npcrflow
export PYTHONPATH=/share/home/lrs/codex/projects/npcrflow/src
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
/share/home/lrs/.conda/envs/mybase/bin/python -u examples/run_pdo_v060_resolution_ablation.py
