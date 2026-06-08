from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

import torch


def _temp_path_for(target: Path) -> Path:
    stamp = time.monotonic_ns() % 1_000_000_000
    return target.with_name(f"_tmp{os.getpid()}_{stamp}{target.suffix}")


def atomic_write_with_retries(
    target: str | Path,
    writer: Callable[[Path], None],
    *,
    attempts: int = 5,
    delay_seconds: float = 0.25,
) -> None:
    """Write to a same-directory temp file, then replace the target with retries."""

    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        temp_path = _temp_path_for(target_path)
        try:
            writer(temp_path)
            temp_path.replace(target_path)
            return
        except (OSError, RuntimeError) as error:
            last_error = error
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            if attempt < attempts - 1:
                time.sleep(delay_seconds * (attempt + 1))
    if last_error is not None:
        raise last_error


def safe_torch_save(
    payload: dict[str, Any],
    target: str | Path,
    *,
    attempts: int = 5,
    delay_seconds: float = 0.25,
) -> None:
    atomic_write_with_retries(
        target,
        lambda temp_path: torch.save(payload, temp_path),
        attempts=attempts,
        delay_seconds=delay_seconds,
    )


def safe_savefig(
    figure,
    target: str | Path,
    *,
    attempts: int = 5,
    delay_seconds: float = 0.25,
) -> None:
    atomic_write_with_retries(
        target,
        lambda temp_path: figure.savefig(temp_path),
        attempts=attempts,
        delay_seconds=delay_seconds,
    )
