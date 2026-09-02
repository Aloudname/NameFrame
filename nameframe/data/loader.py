"""
dataloader factory as core of data processing.

the only method for creating torch `DataLoader` instances.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Sampler

from nameframe.components.batch import Batch
from nameframe.components.dataset import Dataset


def _batch_collate(batches: list[Batch]) -> Batch:
    """
    collates a `list[Batch]` into a single `Batch`.

    stacks `data` and `target` tensors along dim 0.
    merges `meta` `dict` into a list stored under each key.
    if all targets in the batch are `None`,
    set collated target to `None`.

    params:
    - `batches`: `list[Batch]` type.

    returns:
    - `Batch` type of collated batch.
    """
    data = torch.stack([b.data for b in batches])

    targets = [b.target for b in batches]
    if all(t is None for t in targets):
        target = None
    else:
        target = torch.stack([t for t in targets if t is not None])

    # merge per-sample meta dicts into per-key lists
    meta: dict[str, list] = {}
    all_keys: set[str] = set()
    for b in batches:
        all_keys.update(b.meta.keys())
    for key in all_keys:
        meta[key] = [b.meta.get(key) for b in batches]

    return Batch(data=data, target=target, meta=meta)


class DataLoaderFactory:
    """
    core factory creating torch `DataLoader` instances.

    `list[Batch]` -> `Batch`,
    sets defaults for `num_workers` and `prefetch`,
    serves as a single entry for distributed samplers.
    """

    @staticmethod
    def create(
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        sampler: Sampler | None = None,
        drop_last: bool = False,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        **kwargs,
    ) -> DataLoader:
        """
        given `Dataset`, creates a `DataLoader`.

        params:
        - `dataset`: `Dataset` type.
        - `batch_size`: `int` type.
        - `shuffle`: `bool` type, ignored if sampler is provided.
        - `num_workers`: `int` type of occupied threads.
        - `sampler`: optional `Sampler` type, for distributed training.
        - `drop_last`: `bool` type, if to drop last batch when smaller than `batch_size`.
        - `pin_memory`: `bool` type, if to pin memory for faster I/O.
        - `prefetch_factor`: optional `int` type, default 2 when num_workers > 0.

        returns:
        - `DataLoader` instance.
        """
        if prefetch_factor is None and num_workers > 0:
            prefetch_factor = 2

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=_batch_collate,
            drop_last=drop_last,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            **kwargs,
        )
