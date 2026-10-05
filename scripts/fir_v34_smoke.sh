#!/bin/bash
set -euo pipefail

repo_root="${V34_REPO_DIR:-${SLURM_SUBMIT_DIR:-$HOME/portfolio-rcpo-v34}}"
cd "$repo_root"
module load python
venv="${V34_VENV:-$HOME/portfolio-venv}"
if [[ ! -f "$venv/bin/activate" ]]; then
    echo "Missing Python environment: $venv" >&2
    exit 2
fi
source "$venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

config="${1:-configs/v3.4_experiment2_8assets/window_60.yaml}"
updates="${2:-100}"
python scripts/smoke_v34.py --config "$config" --updates "$updates" \
    --output-root profile_outputs/v34_fir_smoke
