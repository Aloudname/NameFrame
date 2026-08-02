"""
`Dataset` follows `DatasetProtocol`,
serves as dataset base class.

[NOTE] `__getitem__` and `__len__` *must* be implemented in subclasses.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset as TorchDataset

from nameframe.components.batch import Batch
from nameframe.utils.typing import FieldSchema


class Dataset(TorchDataset):
    """
    base class for dataset components.

    base class provides `fingerprint()` and `transforms(mode)`.
    [NOTE] `__len__`, `__getitem__` *must* be implemented in subclasses.

    usage::

        from nameframe.registry import dataset
        from nameframe.components import Dataset

        @dataset.register(my_cifar10)
        class MyCIFAR10(Dataset):
            config_schema = {
                "path":    {"type": str, "help": "data root."},
                "augment": {"type": bool, "default": True},
            }

            def __init__(self, path: str, augment: bool = True):
                self.path = path
                self.augment = augment
                ...

            # ↓ something must be implemented ↓

            def __len__(self) -> int: ...
            def __getitem__(self, index: int) -> Batch: ...

            # ↑ something must be implemented ↑

            def _base_transforms(self):
                return [ToTensor(), Normalize(...)]
    """

    config_schema: dict[str, FieldSchema] = {
        "path": {
            "type": str,
            "help": "filesystem path to the dataset root directory.",
        },
        "augment": {
            "type": bool,
            "default": True,
            "help": "if to apply training augmentation.",
        },
    }
    """
    keys:
    - `path`: `str` type of path to dataset root directory.
    - `augment`: `bool` type, if to apply training augmentation.
    """

    _transforms: list[Callable] = []
    """`list[Callable]` type, transforms applied on train set."""


    def fingerprint(self) -> str:
        """
        fingerprint for a dataset.

        combines transforms and dataset path into fingerprint.

        returns:
        - `str` type, the hex digest fingerprint.
        """
        hasher = hashlib.md5()

        # source code of transforms
        for t in self._transforms:
            try:
                src = inspect.getsource(t)
            except (OSError, TypeError):
                src = repr(t)
            hasher.update(src.encode())

        # dataset path
        path = getattr(self, "path", None)
        if path is not None:
            root = Path(path)
            if root.is_dir():
                files = sorted(
                    str(p.relative_to(root))
                    for p in root.rglob("*")
                    if p.is_file()
                )
                for f in files:
                    hasher.update(f.encode())

        return hasher.hexdigest()


    def transforms(self, mode: str = "train") -> list[Callable]:
        """
        returns transforms for the given mode.

        params:
        - `mode`: ``"train"`` or ``"infer"``.

        returns:
        - `list[Callable]` type of transforms.
        """
        base: list[Callable] = self._base_transforms()

        if mode == "infer":
            return list(base)

        if hasattr(self, "augment") and not getattr(self, "augment", True):
            return list(base)

        return base + list(self._transforms)

    def _base_transforms(self) -> list[Callable]:
        """
        returns the non-augmentation transform list.

        optionally override in subclasses.

        the default returns an empty list.

        returns:
        - `list[Callable]` type.
        """
        return []


    # [NOTE] below are the torch dataset contract,
    # which must be implemented in subclasses.
    def __getitem__(self, index: int) -> Batch:
        """
        returns a single data item as a Batch.

        [NOTE] subclasses *must* override this.

        params:
        - `index`: `int` type.

        returns:
        - `Batch` instance.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement __getitem__."
        )

    def __getitems__(self, indices: list[int]) -> list[Batch]:
        """
        Optional, returns a list of data items as a list of Batch.

        [NOTE] *recommendedly*, subclasses override this.

        params:
        - `indices`: `list[int]` type.

        returns:
        - `list[Batch]` type.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement __getitems__."
        )

    def __len__(self) -> int:
        """
        returns the dataset size.

        [NOTE] subclasses *must* override this.

        returns:
        - `int` type.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement __len__."
        )
