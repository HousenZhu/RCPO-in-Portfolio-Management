from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def walltime_hours(value: str) -> float:
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", value)
    if match is None:
        raise ValueError("Walltime must be HH:MM:SS or D-HH:MM:SS")
    days, hours, minutes, seconds = match.groups()
    if int(minutes) >= 60 or int(seconds) >= 60:
        raise ValueError("Walltime minutes and seconds must be below 60")
    return 24 * int(days or 0) + int(hours) + int(minutes) / 60 + int(seconds) / 3600


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Fir walltime against full-size V3.4 smoke estimates.")
    parser.add_argument("walltime")
    parser.add_argument("--smoke-dir", default="profile_outputs/v34_fir_smoke")
    args = parser.parse_args()
    smoke_dir = Path(args.smoke_dir)
    required = (
        "smoke_simplex_v3.4_cf_reward_m2_s5040_e4_runtime_estimate.json",
        "smoke_simplex_v3.4_window_60_runtime_estimate.json",
    )
    projected = []
    for name in required:
        path = smoke_dir / name
        if not path.exists():
            raise SystemExit(f"Missing full-size Fir smoke estimate: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            int(payload["updates"]) < 100
            or int(payload["rollout_steps"]) != 2048
            or int(payload["validation_branches"]) != 20
            or int(payload["test_branches"]) != 50
        ):
            raise SystemExit(f"Smoke does not match production workload: {path}")
        projected.append(float(payload["projected_60000_update_hours_with_30pct_buffer"]))
    requested = walltime_hours(args.walltime)
    minimum = max(projected)
    print(f"Requested {requested:.1f} h; buffered smoke projection {minimum:.1f} h")
    if requested < minimum:
        raise SystemExit("Requested Fir walltime is below the measured 60,000-update projection")


if __name__ == "__main__":
    main()
