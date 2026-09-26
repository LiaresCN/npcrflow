#!/bin/bash
#SBATCH --job-name=npcr-wnpsm-raw
#SBATCH --partition=hujun
#SBATCH --nodelist=c052
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/share/home/lrs/codex/projects/npcrflow/results/wnpsm-raw-final-%j.log

set -euo pipefail
export PYTHONPATH=/share/home/lrs/codex/projects/npcrflow/src
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd /share/home/lrs/codex/projects/npcrflow
/share/home/lrs/.conda/envs/mybase/bin/python -u examples/run_wnpsm_from_raw_dod2k.py
