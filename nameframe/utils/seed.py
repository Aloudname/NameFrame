"""
seed management for reproduce.

provides functions to set global seeds across all random backends
and to derive independent sub-seeds for isolated components.
"""

import hashlib
import json
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
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
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

    hashes all inputs together to avoid deterministic offset/rank collisions.
    returns a stable uint32 seed accepted by numpy and pytorch.

    params:
    - `base_seed`: `int` type, the master seed from set_seed().
    - `component`: `str` type, unique id (e.g. 'model_init', 'dropout').
    - `rank`: `int` type, the distributed process rank. default 0.

    returns:
    - `int` type, the derived seed.
    """
    payload = json.dumps([base_seed, component, rank], ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], byteorder="big")
