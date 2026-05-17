"""Accelerated ops extension layer — ``@accelerated`` decorator system.

Public API::

    from nameframe.ops import accelerated  # the main decorator
    from nameframe.ops import cuda_kernel, triton_kernel, cython_op
    from nameframe.ops import build_all, status
"""

from nameframe.ops.decorators import (
    accelerated,
    build_all,
    cuda_kernel,
    cython_op,
    register_native,
    status,
    triton_kernel,
)

__all__ = [
    "accelerated",
    "cuda_kernel",
    "triton_kernel",
    "cython_op",
    "register_native",
    "build_all",
    "status",
]
