"""
`Batch` follows `BatchProtocol`,
serves as the core for communication between components.

it is frozen after construction,
so components cannot mutate each other's inputs.
the only allowed device transfer creates a new copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class Batch:
    """
    *immutable* data object between components.

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

    def to(self, device: torch.device | str) -> Batch:
        """
        creates a new batch to the given device.
        this is the only device transfer point.

        [NOTE] this method does not mutate the current instance, but returns a new one.

        params:
        - `device`: `torch.device` or `str` type of the target device.

        returns:
        - new `Batch` instance on the given device.
        """
        return Batch(
            data=self.data.to(device),
            target=self.target.to(device) if self.target is not None else None,
            meta=self.meta,  # shallow copy is intentional, for meta is lightweight
        )
