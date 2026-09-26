#!/bin/bash
#SBATCH --job-name=npcr-v034-pdo-lf
#SBATCH --partition=hujun
#SBATCH --nodelist=c061
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/share/home/lrs/codex/projects/npcrflow/results/pdo_v034_reference_scaled_lowpass/slurm-%j.log

set -euo pipefail
cd /share/home/lrs/codex/projects/npcrflow/examples
export PYTHONPATH=/share/home/lrs/codex/projects/npcrflow/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
conda run -n mybase python run_pdo_v034_reference_scaled_lowpass.py
