#!/bin/bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
    echo "Usage: bash scripts/submit_fir_v34.sh WALLTIME [ARRAY_INDICES]" >&2
    echo "Example: bash scripts/submit_fir_v34.sh 4-00:00:00 0-21%22" >&2
    exit 2
fi
walltime="$1"
indices="${2:-0-21%22}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

account="${V34_CPU_ACCOUNT:-}"
if [[ -z "$account" ]]; then
    mapfile -t accounts < <(
        sacctmgr -n -P show assoc where "user=$USER" format=Account,Cluster \
        | awk -F'|' '$2 == "fir" && $1 ~ /_cpu$/ {print $1}' | sort -u
    )
    if (( ${#accounts[@]} != 1 )); then
        echo "Could not identify a unique Fir CPU account. Set V34_CPU_ACCOUNT explicitly." >&2
        exit 2
    fi
    account="${accounts[0]}"
fi
if [[ ! -f "${V34_VENV:-$HOME/portfolio-venv}/bin/activate" ]]; then
    echo "Create/install V34_VENV before submitting." >&2
    exit 2
fi
"${V34_VENV:-$HOME/portfolio-venv}/bin/python" \
    scripts/check_v34_walltime.py "$walltime"
mkdir -p slurm_logs/v3.4
mkdir -p "${V34_RUN_ROOT:-$SCRATCH/portfolio-rcpo-v34-runs}"
common=(--account="$account" --time="$walltime" \
    --array="$indices" --output=slurm_logs/v3.4/v34-%A-%a.out \
    --error=slurm_logs/v3.4/v34-%A-%a.err)
sbatch --test-only "${common[@]}" scripts/fir_v34_array.sh
array_submission="$(sbatch --parsable "${common[@]}" scripts/fir_v34_array.sh)"
array_job="${array_submission%%;*}"
collect_submission="$(sbatch --parsable \
    --dependency="afterany:$array_job" \
    --export="ALL,V34_ARRAY_JOB=$array_job" \
    scripts/fir_v34_collect.sh)"
collect_job="${collect_submission%%;*}"
printf '%s\n' "$array_job" > v34_array_job_id.txt
printf '%s\n' "$collect_job" > v34_collection_job_id.txt
echo "Training array job: $array_job"
echo "Collection job: $collect_job"
