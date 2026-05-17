"""GEMM (General Matrix Multiply-Add) CUDA implementation and fallback."""

from __future__ import annotations

import os

import torch

from nameframe.ops.decorators import accelerated, cuda_kernel

_CUR_DIR: str = os.path.dirname(os.path.abspath(__file__))
_CUDA_SOURCE: str = os.path.normpath(
    os.path.join(_CUR_DIR, "..", "csrc", "gemm.cu")
)

# register CUDA native
@cuda_kernel("gemm", sources=[_CUDA_SOURCE])
def _native_gemm():
    """Placeholder — body is replaced by the JIT-compiled CUDA kernel."""
    pass


# public API
@accelerated("gemm")
def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """General Matrix Multiply-Add: ``D = alpha * (A @ B) + beta * C``.

    When CUDA is available, dispatches to a tiled shared-memory kernel.
    Falls back to :func:`torch.mm` otherwise.

    Args:
        a: Left matrix of shape ``(M, K)``.
        b: Right matrix of shape ``(K, N)``.
        c: Optional addend matrix of shape ``(M, N)``.
           If ``None``, treated as all-zero.
        alpha: Scalar multiplier for ``A @ B``.
        beta: Scalar multiplier for ``C``.

    Returns:
        Result tensor of shape ``(M, N)``.
    """
    import torch

    out: torch.Tensor = torch.mm(a, b)
    out = out * alpha
    if c is not None:
        out = out + beta * c
    return out
