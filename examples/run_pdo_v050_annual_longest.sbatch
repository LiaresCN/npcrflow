#!/bin/bash
#SBATCH --job-name=npcr-v050
#SBATCH --partition=hujun
#SBATCH --nodelist=c061
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/share/home/lrs/codex/projects/npcrflow/logs/pdo_v050_annual_longest-%j.out

set -euo pipefail
cd /share/home/lrs/codex/projects/npcrflow
export PYTHONPATH=/share/home/lrs/codex/projects/npcrflow/src
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
/share/home/lrs/.conda/envs/mybase/bin/python -u examples/run_pdo_v050_annual_longest.py
