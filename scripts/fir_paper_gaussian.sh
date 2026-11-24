#!/bin/bash
#SBATCH --job-name=paper-gaussian
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --output=slurm_logs/paper_gaussian/%A-%a.out
#SBATCH --error=slurm_logs/paper_gaussian/%A-%a.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the portfolio repository}"
module load python
source "${PAPER_VENV:-$HOME/portfolio-venv}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

python scripts/paper_experiments.py \
  --run-root "${PAPER_RUN_ROOT:-$SCRATCH/portfolio-paper-gaussian}" \
  train --task "${SLURM_ARRAY_TASK_ID:?Submit with --array=0-20}"
