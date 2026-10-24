"""Map-style datasets with explicit transform ownership and data versions."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

from torch.utils.data import Dataset as TorchDataset

from nameframe.components.batch import Batch
from nameframe.utils.typing import FieldSchema

TransformFn = Callable[[Batch], Batch]


class Dataset(TorchDataset[Batch]):
    """Subclasses implement raw sample reads and length; transforms run in loaders.

    Class-level _transforms remains a shorthand for stateless functions. Its
    container is copied per instance even when __init__ does not call super().
    Stateful transforms should be created in _augmentation_transforms or
    _base_transforms, which are called once for each new loader.
    """

    config_schema: dict[str, FieldSchema] = {
        "path": {"type": str, "help": "filesystem path to the dataset root."},
        "augment": {"type": bool, "default": True},
    }
    _transforms: Sequence[TransformFn] = ()
    version: str | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> Dataset:
        instance = super().__new__(cls)
        instance._transforms = list(cls._transforms)
        return instance

    def fingerprint(self) -> str:
        """Return an explicit version ID without scanning files or guessing state.

        The owner must change this ID when data, splits or preprocessing change.
        Override this method for an externally computed manifest or content hash.
        """
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError(
                f"{type(self).__name__} must declare a nonempty version or override "
                "fingerprint(); data identity is never inferred from files or repr."
            )
        return self.version

    def transforms(self, mode: str = "train") -> list[TransformFn]:
        """Create a loader's chain: training augmentation, then common processing."""
        if mode not in ("train", "infer"):
            raise ValueError(
                f"unsupported dataset mode '{mode}', expected 'train' or 'infer'."
            )
        base = list(self._base_transforms())
        if mode == "infer" or not getattr(self, "augment", True):
            return base
        return list(self._augmentation_transforms()) + base

    def _augmentation_transforms(self) -> list[TransformFn]:
        """Override to create fresh stateful training transforms per loader."""
        for transform in self._transforms:
            if callable(transform) and not (
                inspect.isfunction(transform) or inspect.isbuiltin(transform)
            ):
                raise TypeError(
                    "callable objects in _transforms can share state; create them "
                    "in _augmentation_transforms() for each loader instead."
                )
        return list(self._transforms)

    def _base_transforms(self) -> list[TransformFn]:
        """Override to create common preprocessing for each new loader."""
        return []

    def __getitem__(self, index: int) -> Batch:
        raise NotImplementedError(f"{type(self).__name__} must implement __getitem__.")

    def __getitems__(self, indices: list[int]) -> list[Batch]:
        """Optional bulk-read optimization; otherwise read each sample once."""
        return [self[index] for index in indices]

    def __len__(self) -> int:
        raise NotImplementedError(f"{type(self).__name__} must implement __len__.")
