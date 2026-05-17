"""Automatic device selection for PyTorch operations."""

from __future__ import annotations

from typing import Optional

import torch


def auto_device(preferred: Optional[str] = None) -> torch.device:
    """Select the best available torch device.

    Priority: explicit *preferred* > CUDA > MPS (Apple Silicon) > CPU.

    Args:
        preferred: Force a specific device string (e.g. ``"cuda:1"``,
            ``"cpu"``). When ``None``, auto-detect the best option.

    Returns:
        A :class:`torch.device` object ready for ``.to(device)`` calls.

    Example:
        >>> device = auto_device()
        >>> model = model.to(device)
    """
    if preferred is not None:
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
