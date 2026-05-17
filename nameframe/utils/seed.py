"""Deterministic random seed initialization across all backends."""

from __future__ import annotations

import random
from typing import Optional


def seed_everything(seed: Optional[int] = None) -> int:
    """Set a deterministic random seed for Python, NumPy, and PyTorch.

    If *seed* is ``None``, a random seed is generated via stdlib.

    Args:
        seed: Integer seed value. If ``None``, a random int is produced.

    Returns:
        The seed value that was actually used.
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
