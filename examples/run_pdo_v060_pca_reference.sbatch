#!/bin/bash
#SBATCH --job-name=npcr-pca-ref
#SBATCH --partition=hujun
#SBATCH --nodelist=c060
#SBATCH --array=0-2
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --output=/share/home/lrs/codex/projects/npcrflow/logs/pdo_v060_pca_reference-%A_%a.out

set -euo pipefail
cd /share/home/lrs/codex/projects/npcrflow
export PYTHONPATH=/share/home/lrs/codex/projects/npcrflow/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
npcr_pca_rules=(kaiser_cv kaiser variance)
/share/home/lrs/.conda/envs/mybase/bin/python -u examples/run_pdo_v060_pca_reference.py "${npcr_pca_rules[$SLURM_ARRAY_TASK_ID]}"
