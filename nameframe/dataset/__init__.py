"""DataLoader factory, builds DataLoaders from registry."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from munch import Munch
from torch.utils.data import DataLoader, Dataset

from nameframe.dataset.base import BaseDataset
from nameframe.registry import DATASET_REGISTRY


def build_dataloader(
    config: Munch,
    split: str,
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    num_workers: Optional[int] = None,
) -> DataLoader[Tuple[torch.Tensor, torch.Tensor]]:
    """Build a single DataLoader for a given data split.

    Looks up ``config.data.dataset`` in :data:`DATASET_REGISTRY` and
    wraps the result in a :class:`DataLoader`.

    Args:
        config: Full project configuration Munch.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        batch_size: Override ``config.train.batch_size``.
        shuffle: Override default shuffle behavior (auto for train).
        num_workers: Override ``config.runtime.num_workers``.

    Returns:
        A PyTorch :class:`DataLoader` for the requested split.
    """
    dataset_name: str = config.data.get("dataset") or config.data.get("name")
    if dataset_name is None:
        raise ValueError(
            "config.data.dataset (or config.data.name) must be set "
            "to a registered dataset name."
        )

    data_subtree: Dict[str, Any] = (
        config.data.toDict() if isinstance(config.data, Munch)
        else dict(config.data)
    )

    dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]] = (
        DATASET_REGISTRY.get_or_build(dataset_name, config=data_subtree, split=split)
    )

    bs: int = batch_size if batch_size is not None else int(config.train.batch_size)
    nw: int = num_workers if num_workers is not None else int(config.runtime.num_workers)
    shuf: bool = shuffle if shuffle is not None else (split == "train")

    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuf,
        num_workers=nw,
        pin_memory=bool(config.runtime.pin_memory),
        drop_last=(split == "train"),
    )


def build_dataloaders(
    config: Munch,
) -> Dict[str, DataLoader[Tuple[torch.Tensor, torch.Tensor]]]:
    """Build train/val/test DataLoaders.

    Args:
        config: Full project configuration Munch.

    Returns:
        Dict mapping split name to DataLoader, e.g.
        ``{"train": ..., "val": ..., "test": ...}``.
    """
    loaders: Dict[str, DataLoader[Tuple[torch.Tensor, torch.Tensor]]] = {}
    for split in ("train", "val", "test"):
        loaders[split] = build_dataloader(config, split)
    return loaders
