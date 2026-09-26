#!/bin/bash
#SBATCH --job-name=npcr-synth-lf
#SBATCH --partition=hujun
#SBATCH --nodelist=c061
#SBATCH --cpus-per-task=18
#SBATCH --mem=36G
#SBATCH --output=/share/home/lrs/codex/projects/npcrflow/results/synthetic_native_window_benchmark/slurm-%j.log

set -euo pipefail
cd /share/home/lrs/codex/projects/npcrflow/examples
export PYTHONPATH=/share/home/lrs/codex/projects/npcrflow/src
export NPCR_SYNTH_WORKERS=18
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
conda run -n mybase python run_synthetic_native_window_benchmark.py
