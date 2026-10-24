"""
dataloader factory as core of data processing.

the only method for creating torch `DataLoader` instances.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data import Dataset as TorchDataset

from nameframe.components.batch import Batch
from nameframe.utils.typing import DatasetProtocol


def _validate_sample(sample: Batch, context: str) -> None:
    """validate the public sample contract without restricting custom collation."""
    if not isinstance(sample, Batch):
        raise TypeError(f"{context} must be a Batch, got {type(sample).__name__}.")
    if not isinstance(sample.data, torch.Tensor):
        raise TypeError(f"{context}.data must be a Tensor.")
    if sample.target is not None and not isinstance(sample.target, torch.Tensor):
        raise TypeError(f"{context}.target must be a Tensor or None.")
    if not isinstance(sample.meta, dict) or any(
        not isinstance(key, str) for key in sample.meta
    ):
        raise TypeError(f"{context}.meta must be a dict with string keys.")


def _stack_field(values: list[torch.Tensor], field: str) -> torch.Tensor:
    """identify the offending sample before asking torch to stack tensors."""
    expected = values[0].shape
    for index, value in enumerate(values[1:], start=1):
        if value.shape != expected:
            raise ValueError(
                f"cannot collate sample[{index}].{field}: shape {tuple(value.shape)} "
                f"differs from sample[0] shape {tuple(expected)}; "
                "provide a custom collate_fn for variable shapes."
            )
        if value.device != values[0].device:
            raise ValueError(
                f"cannot collate sample[{index}].{field}: device {value.device} "
                f"differs from sample[0] device {values[0].device}."
            )
    return torch.stack(values)


def _single_sample_collate(sample: Batch) -> Batch:
    """keep the Batch contract when automatic batching is explicitly disabled."""
    _validate_sample(sample, "sample")
    return sample


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
    if not batches:
        raise ValueError("cannot collate an empty list of samples.")
    for index, batch in enumerate(batches):
        _validate_sample(batch, f"sample[{index}]")

    targets = [b.target for b in batches]
    if all(t is None for t in targets):
        target = None
    elif any(t is None for t in targets):
        raise ValueError(
            "cannot collate samples with mixed present and missing targets; "
            f"missing at sample indices {[i for i, t in enumerate(targets) if t is None]}."
        )
    else:
        target = _stack_field([t for t in targets if t is not None], "target")

    data = _stack_field([b.data for b in batches], "data")

    # merge per-sample meta dicts into per-key lists
    all_keys = dict.fromkeys(key for b in batches for key in b.meta)
    meta = {key: [b.meta.get(key) for b in batches] for key in all_keys}

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
        dataset: DatasetProtocol | TorchDataset,
        batch_size: int | None = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        sampler: Sampler | None = None,
        drop_last: bool = False,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        collate_fn: Callable = _batch_collate,
        batch_sampler: Sampler | None = None,
        persistent_workers: bool = False,
        **kwargs,
    ) -> DataLoader:
        """
        given `Dataset`, creates a `DataLoader`.

        params:
        - `dataset`: `Dataset` type.
        - `batch_size`: positive integer, or None to yield individual samples.
        - `shuffle`: `bool` type, ignored if sampler is provided.
        - `num_workers`: `int` type, number of worker processes.
        - `sampler`: optional `Sampler` type, for distributed training.
        - `drop_last`: `bool` type, if to drop last batch when smaller than `batch_size`.
        - `pin_memory`: `bool` type, if to pin memory for faster I/O.
        - `prefetch_factor`: optional `int` type, default 2 when num_workers > 0.
        - `collate_fn`: callable assembling samples, defaults to Batch collation.
        - `batch_sampler`: owns batching and ordering; ignores shuffle, requires
          the default batch_size=1, no sampler, and drop_last=False.
        - `persistent_workers`: keep worker processes alive between iterations;
          requires num_workers > 0.

        returns:
        - `DataLoader` instance.
        """
        if (
            isinstance(num_workers, bool)
            or not isinstance(num_workers, int)
            or num_workers < 0
        ):
            raise ValueError("num_workers must be a nonnegative integer.")
        if batch_size is not None and (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer or None.")
        if batch_size is None and drop_last:
            raise ValueError("drop_last requires automatic batching (batch_size > 0).")
        if batch_sampler is not None and (
            batch_size != 1 or sampler is not None or drop_last
        ):
            raise ValueError(
                "batch_sampler requires batch_size=1, sampler=None, and drop_last=False; "
                "configure batching on the batch_sampler itself."
            )
        if not isinstance(persistent_workers, bool):
            raise TypeError("persistent_workers must be a bool.")
        if num_workers == 0:
            if persistent_workers:
                raise ValueError("persistent_workers requires num_workers > 0.")
            if kwargs.get("multiprocessing_context") is not None:
                raise ValueError("multiprocessing_context requires num_workers > 0.")
            if kwargs.get("timeout", 0) != 0:
                raise ValueError("timeout must be 0 when num_workers=0.")
        if prefetch_factor is not None and (
            isinstance(prefetch_factor, bool)
            or not isinstance(prefetch_factor, int)
            or prefetch_factor <= 0
        ):
            raise ValueError("prefetch_factor must be a positive integer or None.")
        if prefetch_factor is None and num_workers > 0:
            prefetch_factor = 2

        if num_workers == 0 and prefetch_factor is not None:
            raise ValueError("prefetch_factor requires num_workers > 0.")

        if batch_size is None and collate_fn is _batch_collate:
            collate_fn = _single_sample_collate

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None and batch_sampler is None else False,
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=drop_last,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            batch_sampler=batch_sampler,
            persistent_workers=persistent_workers,
            **kwargs,
        )
