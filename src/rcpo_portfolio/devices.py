from __future__ import annotations

import torch


def resolve_device(device_name: str) -> torch.device:
    normalized = str(device_name).lower().strip()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "runtime.device is set to 'cuda', but CUDA is not available to PyTorch."
            )
        return torch.device("cuda")
    raise ValueError("runtime.device must be one of: auto, cpu, cuda.")


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

