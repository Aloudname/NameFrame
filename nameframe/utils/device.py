"""Device for PyTorch operations."""

from __future__ import annotations

from typing import Optional

import torch


def device(preferred: Optional[str | torch.device] = None) -> torch.device:
    """
    Args:
        preferred: A specific device string (e.g. ``"cuda:1"``,
            ``"cpu"``) or ``torch.device``.

    Returns:
        `torch.device` obj.

    Example:
        >>> device = device()
        >>> model = model.to(device)
    """
    if preferred is not None:
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
