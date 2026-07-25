"""
shared type definitions for the framework.

all component protocols are defined here,
to avoid circular imports between subpackages.

NOTE: 
protocols use structural subtyping via `typing.Protocol`,
so components do not need to explicitly inherit.
"""

from collections.abc import Callable, Iterator
from typing import Any, Protocol, TypedDict

from torch import Tensor
from torch.nn import Parameter


class BatchProtocol(Protocol):
    """protocol for batch data objects passed between components."""

    data: Tensor
    """the model input `torch.Tensor`."""

    # BUG: Union | requires python 3.10+.
    target: Tensor | None
    """the target tensor, or `None` for unsupervised tasks."""

    meta: dict
    """optional metadata `dict` for auxiliary information."""


class ModelProtocol(Protocol):
    """protocol for model components."""

    config_schema: dict
    """declared config schema for the model."""

    def forward(self, batch: BatchProtocol) -> Tensor:
        """
        forward pass.

        params:
        - `batch`: `BatchProtocol` type.

        returns:
        - `Tensor` type, model output.
        """
        ...

    def parameters(self) -> Iterator[Parameter]:
        """returns an iterator over model parameters."""
        ...

    def state_dict(self) -> dict:
        """returns the model's state as a dict."""
        ...

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """
        loads state from a dict.

        params:
        - `state_dict`: `dict` type.
        - `strict`: `bool` type.
        """
        ...


class LossProtocol(Protocol):
    """protocol for loss components."""

    config_schema: dict
    """declared config schema for the loss."""

    def forward(self, pred: Tensor, batch: BatchProtocol) -> Tensor:
        """
        computes the loss.

        params:
        - `pred`: `Tensor` type of model prediction.
        - `batch`: `BatchProtocol` type, the original batch containing targets.

        returns:
        - `Tensor` type, a scalar loss value.
        """
        ...


class MetricProtocol(Protocol):
    """protocol for metric components."""

    config_schema: dict
    """declared config schema for the metric."""

    def update(self, pred: Tensor, batch: BatchProtocol) -> None:
        """
        accumulates a single prediction into the metric state.

        params:
        - `pred`: `Tensor` type, the model prediction.
        - `batch`: `BatchProtocol` type, the original batch.
        """
        ...

    def compute(self) -> dict[str, float]:
        """
        computes the current metric values.

        returns:
        - `dict[str, float]` type, metric name to value mapping.
        """
        ...

    def reset(self) -> None:
        """resets all accumulated metric state."""
        ...


class DatasetProtocol(Protocol):
    """protocol for dataset components."""

    config_schema: dict
    """declared config schema for the dataset."""

    augmentation_transforms: list
    """training-only transforms, auto-disabled during inference."""

    def fingerprint(self) -> str:
        """
        returns a unique fingerprint for the dataset identity.

        returns:
        - `str` type, the fingerprint hash or identifier.
        """
        ...

    def transforms(self, mode: str = "train") -> list[Callable]:
        """
        returns the transform list for the given mode.

        params:
        - `mode`: `str` type, 'train' or 'infer'.

        returns:
        - `list[Callable]` type, the applicable transforms.
        """
        ...

    def __getitem__(self, index: int) -> BatchProtocol:
        """returns a single data item as a BatchProtocol."""
        ...

    def __len__(self) -> int:
        """returns the dataset size."""
        ...


class FieldSchema(TypedDict, total=False):
    """typeddict schema for a single config field."""

    type: type
    """expected `type` type."""

    default: Any
    """`Any` default value."""

    help: str
    """human-readable `str` description of the field."""

    # BUG: Union | requires python 3.10+.
    choices: list | None
    """allowed `list` values, or `None` for no restriction."""

    # BUG: Union | requires python 3.10+.
    range: tuple[float, float] | None
    """allowed `float tuple` range (min, max), or `None`."""

    nullable: bool
    """`bool` if field accepts None. defaults to `false`."""
