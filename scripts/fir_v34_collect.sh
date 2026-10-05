#!/bin/bash
#SBATCH --job-name=v34-collect
#SBATCH --account=def-bereyhia_cpu
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=slurm_logs/v3.4/v34-collect-%j.out
#SBATCH --error=slurm_logs/v3.4/v34-collect-%j.err
set -euo pipefail

umask 027
source_dir="${V34_RUN_ROOT:-$SCRATCH/portfolio-rcpo-v34-runs}"
project_dir="${V34_PROJECT_DIR:-/project/def-bereyhia/$USER/portfolio-rcpo-v34}"
repo_dir="${V34_REPO_DIR:-$HOME/portfolio-rcpo}"

mkdir -p "$project_dir/runs" "$project_dir/slurm_logs" "$project_dir/code"
echo "Copy started: $(date)"
echo "Source: $source_dir"
echo "Destination: $project_dir"
rsync -a --partial "$source_dir/" "$project_dir/runs/"
rsync -a --partial "$repo_dir/slurm_logs/v3.4/" "$project_dir/slurm_logs/"

if [[ -n "${V34_ARRAY_JOB:-}" ]]; then
    sacct -j "$V34_ARRAY_JOB" \
        --format=JobID,JobName%30,State,Elapsed,ReqMem,MaxRSS,ExitCode \
        > "$project_dir/slurm_accounting.txt"
fi
{
    echo "Copy completed: $(date)"
    echo "Training array job: ${V34_ARRAY_JOB:-unknown}"
    echo "Source: $source_dir"
    echo "Destination: $project_dir"
} > "$project_dir/copy_completed.txt"
echo "Copy completed: $(date)"
