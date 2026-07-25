"""
seed management for reproduce.

provides functions to set global seeds across all random backends
and to derive independent sub-seeds for isolated components.
"""

import hashlib
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    sets global random seeds across all supported backends.

    configures random, numpy, torch-cpu, torch-cuda,
    and forces cudnn to a deterministic mode.
    this is the single entry point for making a training run reproducible.

    params:
    - `seed`: `int` type, master seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def derive_seed(base_seed: int, component: str, rank: int = 0) -> int:
    """
    derives an independent seed for a component from a master seed.

    uses an md5 hash of the component name for stable offset.
    this ensures same seed between different components,
    while keeping reproducibility given the same base seed.

    params:
    - `base_seed`: `int` type, the master seed from set_seed().
    - `component`: `str` type, unique id (e.g. 'model_init', 'dropout').
    - `rank`: `int` type, the distributed process rank. default 0.

    returns:
    - `int` type, the derived seed.
    """
    hash_bytes: bytes = hashlib.md5(component.encode()).digest()
    offset: int = int.from_bytes(hash_bytes[:4], byteorder="big") % 10000
    return base_seed + offset + rank * 1000
