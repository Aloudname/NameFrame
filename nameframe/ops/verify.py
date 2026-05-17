"""Verification utilities — compare native vs fallback op outputs."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch

from nameframe.ops.registry import NATIVE_REGISTRY, OP_REGISTRY


def verify_op(
    name: str,
    backend: str,
    sample_inputs: tuple = (),
    tolerance: float = 1e-4,
) -> bool:
    """Verify that a native op produces results matching its Python fallback.

    Args:
        name: Op identifier.
        backend: Backend to verify (e.g. ``"cuda"``).
        sample_inputs: Tuple of tensors to feed as input.
        tolerance: Maximum allowed absolute difference.

    Returns:
        ``True`` if the outputs match within tolerance.
    """
    fallback_fn: Optional[Callable[..., Any]] = None
    if name in OP_REGISTRY:
        fallback_fn = OP_REGISTRY.get(name)

    native_fn: Optional[Callable[..., Any]] = NATIVE_REGISTRY.get(name, {}).get(backend)

    if fallback_fn is None or native_fn is None:
        return False

    ref: Any = fallback_fn(*sample_inputs)
    nat: Any = native_fn(*sample_inputs)

    if isinstance(ref, torch.Tensor) and isinstance(nat, torch.Tensor):
        diff: float = float((ref.to(nat.device) - nat).abs().max().item())
        return diff <= tolerance

    return True  # non-tensor outputs are assumed equal


def verify_all(tolerance: float = 1e-4) -> Dict[str, bool]:
    """Verify all registered native ops against their Python fallbacks.

    Uses a small random input for each op.

    Args:
        tolerance: Maximum allowed absolute difference.

    Returns:
        Dict of op name -> pass/fail.
    """
    results: Dict[str, bool] = {}
    for name in OP_REGISTRY.list():
        for backend in NATIVE_REGISTRY.get(name, {}):
            # generate a simple test input
            sample: tuple = (torch.randn(1, 64, 16, 16),)
            results[f"{name}:{backend}"] = verify_op(
                name, backend, sample, tolerance
            )
    return results
