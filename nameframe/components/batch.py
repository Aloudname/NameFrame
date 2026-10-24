"""
serves as the core for communication between components.

fields cannot be rebound after construction, tensors and metadata stay mutable.
components should return new results instead of mutating shared inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class Batch:
    """
    data object with frozen field bindings between components.

    params:
    - `data`: `Tensor` type of model input tensor.
    - `target`: optional `Tensor` type of the ground truth. `None`
      for unsupervised or inference-only tasks.
    - `meta`: `dict` type, optional metadata.
    """

    data: Tensor
    """model input `Tensor`."""

    target: Tensor | None = None
    """ground truth `Tensor`, or `None`."""

    meta: dict[str, Any] = field(default_factory=dict)
    """metadata `dict`."""

    def to(self, device: torch.device | str, non_blocking: bool = False) -> Batch:
        """
        creates a new batch with data and target on the given device.
        same-device tensor transfers may reuse existing storage.
        metadata remains host-side and is shallow-copied.

        [NOTE] this method does not mutate the current instance, but returns a new one.

        params:
        - `device`: `torch.device` or `str` type of the target device.

        returns:
        - new `Batch` instance on the given device.
        """
        return Batch(
            data=self.data.to(device, non_blocking=non_blocking),
            target=(
                self.target.to(device, non_blocking=non_blocking)
                if self.target is not None
                else None
            ),
            meta=dict(self.meta),
        )

    def pin_memory(self) -> Batch:
        """return pinned tensors so pytorch can pin this custom batch type."""
        return Batch(
            data=self.data.pin_memory(),
            target=self.target.pin_memory() if self.target is not None else None,
            meta=dict(self.meta),
        )
