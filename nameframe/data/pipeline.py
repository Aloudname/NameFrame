"""
data pipeline.

wraps `Dataset` with its transforms and dataloader into a single entry.
pipeline does not need to know what kind of data flows through it.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from nameframe.components.dataset import Dataset
from nameframe.data.loader import DataLoaderFactory


class DataPipeline:
    """
    wraps dataset, transforms and dataloader.

    params:
    - `dataset`: `Dataset` type.
    - `batch_size`: `int` type.
    - `mode`: `str` type, `"train"` or `"infer"`.
    - `shuffle`: `bool` type.
    - `num_workers`: `int` type.
    - `**loader_kwargs`: kwargs to `DataLoader`.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        mode: str = "train",
        shuffle: bool = True,
        num_workers: int = 0,
        **loader_kwargs,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.mode = mode
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.loader_kwargs = loader_kwargs

    def get_loader(self) -> DataLoader:
        """
        given `Dataset`, creates a torch `DataLoader`.

        returns:
        - `torch.utils.data.DataLoader` instance.
        """
        return DataLoaderFactory.create(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            **self.loader_kwargs,
        )

    def __iter__(self):
        """iterates over the loader created by `get_loader()`."""
        return iter(self.get_loader())
