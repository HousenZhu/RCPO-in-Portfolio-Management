from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import torch


@contextmanager
def exclusive_run_lock(run_dir: str | Path) -> Iterator[None]:
    """Prevent multiple trainers from writing the same run directory."""

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    lock_path = run_path / ".training.lock"
    handle = lock_path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(
                f"Another training process is already using run directory: {run_path}"
            ) from error
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


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


def live_savefig(
    figure,
    target: str | Path,
    *,
    attempts: int = 5,
    delay_seconds: float = 0.25,
) -> None:
    """Overwrite a live plot in place so Windows viewers can observe the update."""

    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            figure.savefig(target_path)
            return
        except (OSError, RuntimeError) as error:
            last_error = error
            if attempt < attempts - 1:
                time.sleep(delay_seconds * (attempt + 1))
    if last_error is not None:
        raise last_error
