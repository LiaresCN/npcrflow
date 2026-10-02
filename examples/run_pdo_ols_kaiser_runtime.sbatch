#!/bin/bash
#SBATCH --job-name=npcr-ols-kaiser
#SBATCH --partition=hujun
#SBATCH --nodelist=c061
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/share/home/lrs/codex/projects/npcrflow/logs/pdo_ols_kaiser_runtime-%j.out

set -euo pipefail
cd /share/home/lrs/codex/projects/npcrflow
export PYTHONPATH=/share/home/lrs/codex/projects/npcrflow/src
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
/share/home/lrs/.conda/envs/mybase/bin/python examples/run_pdo_ols_kaiser_runtime.py
