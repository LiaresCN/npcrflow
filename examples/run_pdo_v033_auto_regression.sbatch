#!/bin/bash
#SBATCH --job-name=npcr-v033-auto
#SBATCH --partition=hujun
#SBATCH --nodelist=c061
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/share/home/lrs/codex/projects/npcrflow/results/pdo_v033_auto_regression/slurm-%j.log

set -euo pipefail
cd /share/home/lrs/codex/projects/npcrflow/examples
export PYTHONPATH=/share/home/lrs/codex/projects/npcrflow/src
conda run -n mybase python run_pdo_v033_auto_regression.py
