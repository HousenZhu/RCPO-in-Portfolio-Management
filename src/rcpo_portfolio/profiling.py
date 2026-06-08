from __future__ import annotations

import csv
import json
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch


@dataclass
class TimingRecord:
    count: int = 0
    total_seconds: float = 0.0

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / max(self.count, 1)


class TrainingProfiler:
    """Lightweight wall-clock profiler for noisy training-loop diagnostics."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.device = torch.device(device) if device is not None else None
        self.records: dict[str, TimingRecord] = {}
        self._started_at = time.perf_counter()

    def _synchronize(self) -> None:
        if (
            self.enabled
            and self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(self.device)

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._synchronize()
        started_at = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            elapsed = time.perf_counter() - started_at
            record = self.records.setdefault(name, TimingRecord())
            record.count += 1
            record.total_seconds += elapsed

    def total_runtime_seconds(self) -> float:
        return max(time.perf_counter() - self._started_at, 0.0)

    def sorted_rows(self) -> list[dict[str, Any]]:
        runtime = max(self.total_runtime_seconds(), 1e-12)
        rows = [
            {
                "section": name,
                "count": record.count,
                "total_seconds": record.total_seconds,
                "mean_seconds": record.mean_seconds,
                "percent_of_runtime": 100.0 * record.total_seconds / runtime,
            }
            for name, record in self.records.items()
        ]
        return sorted(rows, key=lambda row: row["total_seconds"], reverse=True)

    def summary(self) -> dict[str, Any]:
        rows = self.sorted_rows()
        return {
            "enabled": self.enabled,
            "device": str(self.device) if self.device is not None else None,
            "total_runtime_seconds": self.total_runtime_seconds(),
            "sections": {row["section"]: row for row in rows},
        }

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2)

    def write_csv(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = self.sorted_rows()
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "section",
                    "count",
                    "total_seconds",
                    "mean_seconds",
                    "percent_of_runtime",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def format_table(self, *, limit: int | None = None) -> str:
        rows = self.sorted_rows()
        if limit is not None:
            rows = rows[:limit]
        lines = [
            f"{'section':36s} {'count':>8s} {'total_s':>10s} "
            f"{'mean_s':>10s} {'runtime_%':>10s}"
        ]
        for row in rows:
            lines.append(
                f"{row['section']:36s} {row['count']:8d} "
                f"{row['total_seconds']:10.4f} {row['mean_seconds']:10.6f} "
                f"{row['percent_of_runtime']:10.2f}"
            )
        return "\n".join(lines)


def profile_section(
    profiler: TrainingProfiler | None,
    name: str,
):
    if profiler is None:
        return nullcontext()
    return profiler.section(name)
