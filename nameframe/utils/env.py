"""
runtime env capture for reproducibility diagnostics.

records the software stack signature,
so that experiments can be reproduced in equivalent environments.
"""

import platform
import sys

import torch


def capture_env() -> dict:
    """
    captures the runtime environment signature.

    returns a dict with keys 'python', 'torch', 'cuda', 'cudnn', and
    'platform'. cuda and cudnn report 'none' when unavailable.

    returns:
    - `dict` type, mapping component names to version strings.
    """
    cuda_version: str
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda or "unknown"  # type: ignore
    else:
        cuda_version = "none"

    cudnn_version: str
    version = torch.backends.cudnn.version() if torch.cuda.is_available() else None
    cudnn_version = str(version) if version is not None else "none"

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": cuda_version,
        "cudnn": cudnn_version,
        "platform": platform.platform(),
    }
