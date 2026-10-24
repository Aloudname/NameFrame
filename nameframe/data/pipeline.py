"""compose per-sample batch transforms and loading without mutating datasets."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from nameframe.components.batch import Batch
from nameframe.data.loader import DataLoaderFactory, _validate_sample
from nameframe.utils.typing import DatasetProtocol


class _TransformedDataset(TorchDataset):
    """loader-local view, each raw sample passes through the chain exactly once."""

    def __init__(
        self, dataset: DatasetProtocol, transforms: list[Callable[[Batch], Batch]]
    ) -> None:
        self.dataset = dataset
        self.transforms = list(transforms)
        for position, transform in enumerate(self.transforms):
            if not callable(transform):
                raise TypeError(f"transform[{position}] must be callable.")

    def __len__(self) -> int:
        return len(self.dataset)

    def _apply(self, batch: Batch, index: int) -> Batch:
        context = f"{type(self.dataset).__name__}[{index}]"
        _validate_sample(batch, context)
        for position, transform in enumerate(self.transforms):
            batch = transform(batch)
            name = getattr(transform, "__qualname__", type(transform).__qualname__)
            _validate_sample(batch, f"{context} after transform[{position}] {name}")
        return batch

    def __getitem__(self, index: int) -> Batch:
        return self._apply(self.dataset[index], index)

    def __getitems__(self, indices: list[int]) -> list[Batch]:
        fetch = getattr(self.dataset, "__getitems__", None)
        samples = (
            fetch(indices) if callable(fetch) else [self.dataset[i] for i in indices]
        )
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
            raise TypeError(
                f"{type(self.dataset).__name__}.__getitems__ must return a sequence "
                "of Batch samples."
            )
        if len(samples) != len(indices):
            raise ValueError(
                f"{type(self.dataset).__name__}.__getitems__ returned {len(samples)} "
                f"samples for {len(indices)} requested indices."
            )
        return [
            self._apply(sample, index)
            for sample, index in zip(samples, indices, strict=True)
        ]


class DataPipeline:
    """apply dataset.transforms(mode) before collation, keep transfers external.

    dataset.__getitem__ returns an untransformed Batch containing tensor data.
    transform callables accept and return a per-sample Batch, allowing synchronized
    changes to data, target, and metadata. infer mode disables augmentation and
    defaults to stable sample order. each loader owns its dataset view, so train
    and validation loaders can safely share a dataset with different modes.
    """

    def __init__(
        self,
        dataset: DatasetProtocol,
        batch_size: int | None = 1,
        mode: str = "train",
        shuffle: bool | None = None,
        num_workers: int = 0,
        **loader_kwargs: Any,
    ) -> None:
        if mode not in ("train", "infer"):
            raise ValueError(
                f"unsupported pipeline mode '{mode}', expected 'train' or 'infer'."
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.mode = mode
        self.shuffle = mode == "train" if shuffle is None else shuffle
        self.num_workers = num_workers
        self.loader_kwargs = loader_kwargs

    def get_loader(self) -> DataLoader[Batch]:
        """create a loader with its own mode-specific transform view."""
        dataset = _TransformedDataset(self.dataset, self.dataset.transforms(self.mode))
        return DataLoaderFactory.create(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            **self.loader_kwargs,
        )

    def __iter__(self) -> Iterator[Batch]:
        """iterate through a newly constructed loader."""
        return iter(self.get_loader())
