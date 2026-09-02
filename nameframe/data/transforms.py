"""
transform base class for data processing.

transforms are `Callable` that accept and return a `Batch`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from nameframe.components.batch import Batch


class Transform(ABC):
    """
    base class for all data transforms.

    transforms are `Callable` take and return a `Batch`.
    subclasses must implement `__call__`.
    """

    @abstractmethod
    def __call__(self, batch: Batch) -> Batch:
        """
        applies the transform to a `Batch`.

        params:
        - `batch`: `Batch` type of input batch.

        returns:
        - `Batch` type of transformed batch.
        """
        ...
