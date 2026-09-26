#!/bin/bash
#SBATCH --job-name=npcr-pdo-core
#SBATCH --partition=hujun
#SBATCH --nodelist=c061
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/share/home/lrs/codex/projects/npcrflow/results/pdo-two-thirds-core-%j.log

set -euo pipefail
export PYTHONPATH=/share/home/lrs/codex/projects/npcrflow/src
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd /share/home/lrs/codex/projects/npcrflow
/share/home/lrs/.conda/envs/mybase/bin/python -u examples/run_pdo_from_raw_dod2k.py
