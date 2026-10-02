#!/bin/bash
#SBATCH --job-name=npcr-v060
#SBATCH --partition=hujun
#SBATCH --nodelist=c061
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/share/home/lrs/codex/projects/npcrflow/logs/pdo_v060_explicit_nests-%j.out

set -euo pipefail
cd /share/home/lrs/codex/projects/npcrflow
export PYTHONPATH=/share/home/lrs/codex/projects/npcrflow/src
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
/share/home/lrs/.conda/envs/mybase/bin/python -u examples/run_pdo_v060_explicit_nests.py "$@"
