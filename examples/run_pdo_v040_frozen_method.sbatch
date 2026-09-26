#!/bin/bash
#SBATCH --job-name=npcr-v040-freeze
#SBATCH --partition=hujun
#SBATCH --nodelist=c060
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/share/home/lrs/codex/projects/npcrflow/results/pdo-v040-freeze-%j.log

set -euo pipefail
export PYTHONPATH=/share/home/lrs/codex/projects/npcrflow/src
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
cd /share/home/lrs/codex/projects/npcrflow
/share/home/lrs/.conda/envs/mybase/bin/python -u examples/run_pdo_v040_frozen_method.py
