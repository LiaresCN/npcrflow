#!/bin/bash
#SBATCH --job-name=npcr-pca-v041
#SBATCH --partition=hujun
#SBATCH --nodelist=c060
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/share/home/lrs/codex/projects/npcrflow/results/pdo-v041-pca-%j.log

set -euo pipefail
cd /share/home/lrs/codex/projects/npcrflow
export PYTHONPATH=/share/home/lrs/codex/projects/npcrflow/src
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
/share/home/lrs/.conda/envs/mybase/bin/python -u examples/run_pdo_v041_pca_selection.py
