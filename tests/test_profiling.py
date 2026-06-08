from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from rcpo_portfolio.profiling import TrainingProfiler


def test_training_profiler_records_serializable_sections() -> None:
    profiler = TrainingProfiler(enabled=True, device="cpu")
    with profiler.section("rollout_total"):
        pass
    with profiler.section("rollout_total"):
        pass
    with profiler.section("env_step"):
        pass

    summary = profiler.summary()
    assert summary["enabled"] is True
    assert summary["sections"]["rollout_total"]["count"] == 2
    assert summary["sections"]["env_step"]["count"] == 1
    assert summary["sections"]["rollout_total"]["total_seconds"] >= 0.0
    assert summary["sections"]["rollout_total"]["mean_seconds"] >= 0.0
    assert summary["sections"]["rollout_total"]["percent_of_runtime"] >= 0.0
    json.dumps(summary)


def test_profile_training_script_smoke(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / ".pytest_cache" / "profile_smoke"
    script_path = project_root / "scripts" / "profile_training.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--algo",
            "ppo_unconstrained",
            "--updates",
            "1",
            "--rollout-steps",
            "32",
            "--epochs",
            "1",
            "--minibatch-size",
            "16",
            "--output-root",
            str(output_root),
            "--no-write-plots",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "rollout_total" in result.stdout
    summary_path = output_root / "profile_summary.json"
    timing_path = (
        output_root / "ppo_unconstrained_simplex_branch_gaussian_timing.json"
    )
    assert summary_path.exists()
    assert timing_path.exists()
    with timing_path.open("r", encoding="utf-8") as handle:
        timing = json.load(handle)
    sections = timing["sections"]
    for key in [
        "rollout_total",
        "optimization_total",
        "policy_action_forward",
        "env_step",
        "model_recompute_forward",
        "loss_compute",
        "backward",
        "optimizer_step",
        "metric_write",
        "checkpoint_last",
    ]:
        assert key in sections
        assert sections[key]["total_seconds"] >= 0.0
