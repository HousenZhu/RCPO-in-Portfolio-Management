# V3.4 Fir operation guide

This guide follows the V3.3 Fir workflow. The 22 new runs are defined in
`tasks.tsv`; array indices `0-21` correspond to its 22 data rows. All 22 tasks
may be submitted concurrently with `--array=0-21%22`. Slurm can still leave
some tasks pending when Fir does not have enough available cores.

Each training task requests one CPU core and 6 GB of memory. Training output is
written to Scratch, then an automatically dependent collection job copies the
results and Slurm accounting records to Project storage.

## 1. Enter the existing Fir project environment

The following paths match the V3.3 setup already used on Fir:

```bash
cd "$HOME/portfolio-rcpo"
module load python
source "$HOME/portfolio-venv/bin/activate"

SCRATCH_RUNS="$SCRATCH/portfolio-rcpo-v34-runs"
PROJECT_DIR="/project/def-bereyhia/$USER/portfolio-rcpo-v34"

mkdir -p "$SCRATCH_RUNS"
mkdir -p "$PROJECT_DIR/code" "$PROJECT_DIR/runs" "$PROJECT_DIR/slurm_logs"
mkdir -p "$HOME/portfolio-rcpo/slurm_logs/v3.4"
```

Confirm the CPU account and the 22-task manifest:

```bash
sacctmgr -n -P show assoc where "user=$USER" format=Account,Cluster
wc -l configs/v3.4_experiment2_8assets/tasks.tsv
column -t -s $'\t' configs/v3.4_experiment2_8assets/tasks.tsv
```

`wc -l` should report `23`: one header plus 22 tasks. The supplied scripts use
the existing Fir CPU account `def-bereyhia_cpu`.

## 2. Check scripts and required Slurm settings

```bash
bash -n scripts/fir_v34_array.sh
bash -n scripts/fir_v34_collect.sh
bash -n scripts/submit_fir_v34.sh

grep '^#SBATCH' scripts/fir_v34_array.sh
grep '^#SBATCH' scripts/fir_v34_collect.sh
```

The training script must show:

```text
partition=cpubase_bycore_b5
time=4-00:00:00
cpus-per-task=1
mem=6G
array=0-21%22
```

The job-array script writes each run below:

```text
$SCRATCH/portfolio-rcpo-v34-runs/<task_id>/
```

If an array task is submitted again, it resumes an unfinished run from
`checkpoint_last.pt` until the total reaches 60,000 updates. A completed task
is skipped rather than overwritten.

## 3. Run the representative 100-update Fir smoke tests

Run the counterfactual and longest-window cases before the full array:

```bash
sbatch --account=def-bereyhia_cpu \
  --partition=cpubase_bycore_b5 \
  --time=04:00:00 \
  --cpus-per-task=1 \
  --mem=6G \
  --output=slurm_logs/v3.4/smoke-%j.out \
  scripts/fir_v34_smoke.sh \
  configs/v3.4_experiment2_8assets/cf_reward_m2_e4.yaml 100

sbatch --account=def-bereyhia_cpu \
  --partition=cpubase_bycore_b5 \
  --time=04:00:00 \
  --cpus-per-task=1 \
  --mem=6G \
  --output=slurm_logs/v3.4/smoke-%j.out \
  scripts/fir_v34_smoke.sh \
  configs/v3.4_experiment2_8assets/window_60.yaml 100
```

Inspect the smoke outputs and runtime estimates:

```bash
sacct -j SMOKE_JOB_ID --format=JobID,State,ExitCode,Elapsed,ReqMem,MaxRSS
tail -n 40 slurm_logs/v3.4/smoke-SMOKE_JOB_ID.out
ls -lh profile_outputs/v34_fir_smoke
```

The formal four-day walltime is valid only if both buffered 60,000-update
projections fit within it:

```bash
python scripts/check_v34_walltime.py 4-00:00:00
```

## 4. Submit all 22 training tasks simultaneously

The recommended command performs a dry-run validation, submits the array with
`sbatch --array=0-21%22`, and schedules the result collection job:

```bash
bash scripts/submit_fir_v34.sh 4-00:00:00 0-21%22
```

The defaults are also `0-21%22`, so this shorter command is equivalent:

```bash
bash scripts/submit_fir_v34.sh 4-00:00:00
```

The helper records the returned IDs in:

```text
v34_array_job_id.txt
v34_collection_job_id.txt
```

Internally, the training submission uses the equivalent Slurm setting:

```bash
sbatch --array=0-21%22 scripts/fir_v34_array.sh
```

The collection task uses `afterany`, so it runs after every array element has
finished even if one element fails. It copies results to:

```text
/project/def-bereyhia/$USER/portfolio-rcpo-v34
```

## 5. Monitor training

```bash
ARRAY_JOB=$(cat v34_array_job_id.txt)
COLLECT_JOB=$(cat v34_collection_job_id.txt)

squeue -j "$ARRAY_JOB","$COLLECT_JOB" \
  -o "%.18i %.24j %.24P %.10T %.12M %.12l %R"

sacct -j "$ARRAY_JOB" \
  --format=JobID,JobName%25,Partition%24,State,Elapsed,ReqMem,MaxRSS,ExitCode

tail -n 40 "slurm_logs/v3.4/v34-${ARRAY_JOB}-0.out"
du -sh "$SCRATCH/portfolio-rcpo-v34-runs"
find "$SCRATCH/portfolio-rcpo-v34-runs" -name checkpoint_last.pt | wc -l
```

Expected behavior:

- Up to 22 array elements can be `RUNNING` at once.
- Fir may keep some elements `PENDING` because `%22` is a concurrency ceiling,
  not a resource guarantee.
- The collection job remains `PENDING (Dependency)` until the array ends.
- Each task has an independent log and Scratch output directory.

To retry only selected failed elements without touching completed tasks:

```bash
bash scripts/submit_fir_v34.sh 4-00:00:00 3,8
```

## 6. Verify the collected results

```bash
PROJECT_DIR="/project/def-bereyhia/$USER/portfolio-rcpo-v34"

cat "$PROJECT_DIR/copy_completed.txt"
cat "$PROJECT_DIR/slurm_accounting.txt"
du -sh "$PROJECT_DIR"
find "$PROJECT_DIR/runs" -name training_summary.json | wc -l
grep -E "FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY" \
  "$PROJECT_DIR/slurm_accounting.txt"
```

All-new training should produce 22 `training_summary.json` files. An empty
`grep` result means no matching failure state was found.

## 7. Download and compare results

Create the archive on Fir:

```bash
cd "/project/def-bereyhia/$USER"
tar -czf portfolio-rcpo-v34-results.tar.gz portfolio-rcpo-v34
```

Download from Windows PowerShell:

```powershell
scp housen@fir.alliancecan.ca:/project/def-bereyhia/housen/portfolio-rcpo-v34-results.tar.gz .
tar -xzf .\portfolio-rcpo-v34-results.tar.gz
```

After placing the completed run folders under the local `runs/v3.4/` layout,
evaluate best-return and available best-feasible checkpoints on three shared
holdout anchors with 20 future branches each:

```powershell
py -3.11 scripts/compare_v34.py
```

Comparison outputs are written to `evaluation/v3.4_policy_comparison/` and
include both each model's own drawdown rule and the common 0.90-margin risk
rule.
