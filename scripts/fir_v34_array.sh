#!/bin/bash
#SBATCH --job-name=v34
#SBATCH --account=def-bereyhia_cpu
#SBATCH --partition=cpubase_bycore_b5
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --array=0-21%22
#SBATCH --output=slurm_logs/v3.4/v34-%A-%a.out
#SBATCH --error=slurm_logs/v3.4/v34-%A-%a.err
set -euo pipefail

repo_root="${V34_REPO_DIR:-${SLURM_SUBMIT_DIR:-$HOME/portfolio-rcpo-v34}}"
cd "$repo_root"
manifest="configs/v3.4_experiment2_8assets/tasks.tsv"
row="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$manifest")"
if [[ -z "$row" ]]; then
    echo "No V3.4 task for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
IFS=$'\t' read -r task_id group config_name seed <<< "$row"
config="configs/v3.4_experiment2_8assets/$config_name"
run_root="${V34_RUN_ROOT:-$SCRATCH/portfolio-rcpo-v34-runs}"
output_root="$run_root/$task_id"

module load python
venv="${V34_VENV:-$HOME/portfolio-venv}"
if [[ ! -f "$venv/bin/activate" ]]; then
    echo "Missing Python environment: $venv" >&2
    exit 2
fi
source "$venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
mkdir -p "$output_root"

shopt -s nullglob
existing=("$output_root"/*/"seed_$seed")
if (( ${#existing[@]} > 1 )); then
    echo "Multiple run directories for $task_id; refusing ambiguous resume" >&2
    exit 2
fi
echo "V3.4 task=$task_id group=$group seed=$seed config=$config"
if (( ${#existing[@]} == 1 )); then
    if [[ -f "${existing[0]}/training_summary.json" ]]; then
        echo "Already complete: ${existing[0]}"
        exit 0
    fi
    if [[ ! -f "${existing[0]}/checkpoint_last.pt" ]]; then
        echo "Run exists without checkpoint_last.pt: ${existing[0]}" >&2
        exit 2
    fi
    python train.py --algo rcpo --constraint-drawdown \
        --resume-run-dir "${existing[0]}" --resume-checkpoint checkpoint_last.pt \
        --resume-target-total-updates 60000
else
    python train.py --algo rcpo --constraint-drawdown \
        --config "$config" --seed "$seed" --output-root "$output_root"
fi
