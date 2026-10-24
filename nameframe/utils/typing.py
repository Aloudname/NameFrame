"""Public component contracts; configuration metadata is optional."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from torch import Tensor

if TYPE_CHECKING:
    from nameframe.components.batch import Batch


class ModelProtocol(Protocol):
    """A model consumes the same concrete Batch as the data pipeline."""

    def __call__(self, batch: Batch) -> Tensor: ...


class LossProtocol(Protocol):
    """A loss consumes predictions and the original batch."""

    def __call__(self, pred: Tensor, batch: Batch) -> Tensor: ...


class MetricProtocol(Protocol):
    """Metric state belongs to the caller's metric instance."""

    def update(self, pred: Tensor, batch: Batch) -> None: ...
    def compute(self) -> dict[str, float]: ...
    def reset(self) -> None: ...


class DatasetProtocol(Protocol):
    """Only the capabilities consumed by DataPipeline are required."""

    def transforms(self, mode: str = "train") -> list[Callable[[Batch], Batch]]: ...
    def __getitem__(self, index: int) -> Batch: ...
    def __len__(self) -> int: ...


class FieldSchema(TypedDict, total=False):
    """A missing default makes the field required; values are not coerced."""

    type: type
    default: Any
    help: str
    choices: list | None
    range: tuple[float, float] | None
    nullable: bool


def _field(
    type: type | None = None,
    default: Any = ...,
    help: str | None = None,
    choices: list | None = None,
    range: tuple[float, float] | None = None,
    nullable: bool = False,
) -> FieldSchema:
    """Build a field definition without adding omitted rules."""
    result: FieldSchema = {}
    if type is not None:
        result["type"] = type
    if default is not ...:
        result["default"] = default
    if help is not None:
        result["help"] = help
    if choices is not None:
        result["choices"] = choices
    if range is not None:
        result["range"] = range
    if nullable:
        result["nullable"] = nullable
    return result
