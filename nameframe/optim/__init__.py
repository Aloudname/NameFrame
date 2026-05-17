"""Optimizer and learning-rate scheduler factories."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from munch import Munch


def build_optimizer(model: nn.Module, config: Munch) -> optim.Optimizer:
    """Build an optimizer from config.

    Supports ``adamw``, ``sgd``, ``adam``. Handles per-layer learning-rate
    multipliers via ``config.optim.param_groups``.

    Args:
        model: The model whose parameters will be optimized.
        config: Full project config Munch.

    Returns:
        A PyTorch :class:`Optimizer` instance.
    """
    opt_name: str = config.optim.get("name", "adamw").lower()
    lr: float = float(config.train.lr)
    wd: float = float(config.train.weight_decay)
    betas: tuple[float, float] = tuple(config.optim.get("betas", [0.9, 0.999]))
    eps: float = float(config.optim.get("eps", 1e-8))
    param_groups: List[Dict[str, Any]] = _resolve_param_groups(
        model, lr, config.optim.get("param_groups", [])
    )

    kwargs: Dict[str, Any] = {"lr": lr, "weight_decay": wd}
    if not param_groups:
        param_groups = [{"params": model.parameters()}]

    if opt_name == "adamw":
        return optim.AdamW(param_groups, betas=betas, eps=eps, **kwargs)
    elif opt_name == "sgd":
        momentum: float = float(config.optim.get("momentum", 0.9))
        return optim.SGD(param_groups, momentum=momentum, **kwargs)
    elif opt_name == "adam":
        return optim.Adam(param_groups, betas=betas, eps=eps, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(
    optimizer: optim.Optimizer,
    config: Munch,
    steps_per_epoch: int = 1,
) -> object:
    """Build a learning-rate scheduler from configuration.

    Supported types: ``cosine``, ``step``, ``poly``, ``plateau``,
    ``onecycle``, ``warmup_cosine``.

    Args:
        optimizer: The optimizer to schedule.
        config: Full project configuration Munch.
        steps_per_epoch: Number of optimizer steps per epoch (used by
            ``onecycle`` and ``warmup_cosine`` for total step calculation).

    Returns:
        A PyTorch scheduler object (e.g. :class:`LRScheduler` or
        :class:`ReduceLROnPlateau`).
    """
    sched_name: str = config.scheduler.get("name", "cosine").lower()
    epochs: int = int(config.train.epochs)
    warmup: int = int(config.scheduler.get("warmup_epochs", 5))
    min_lr: float = float(config.scheduler.get("min_lr", 1e-6))

    if sched_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    elif sched_name == "step":
        step_size: int = int(config.scheduler.get("step_size", 30))
        gamma: float = float(config.scheduler.get("step_gamma", 0.1))
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif sched_name == "poly":
        def _poly_lambda(epoch: int) -> float:
            return (1.0 - epoch / epochs) ** 0.9
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_poly_lambda)

    elif sched_name == "plateau":
        patience: int = int(config.scheduler.get("plateau_patience", 5))
        factor: float = float(config.scheduler.get("plateau_factor", 0.5))
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience
        )

    elif sched_name == "onecycle":
        total_steps: int = epochs * steps_per_epoch
        max_lr: float = float(config.train.lr)
        return optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps
        )

    elif sched_name == "warmup_cosine":
        total_steps: int = epochs * steps_per_epoch
        warmup_steps: int = warmup * steps_per_epoch

        def _warmup_cosine(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress: float = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_cosine)

    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


def _resolve_param_groups(
    model: nn.Module,
    base_lr: float,
    groups_cfg: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert named per-layer LR multiplier config into param_group dicts.

    Args:
        model: The model.
        base_lr: Base learning rate.
        groups_cfg: List of ``{name: str, lr_mult: float}`` dicts.

    Returns:
        List of param_group dicts for the optimizer.
    """
    if not groups_cfg:
        return [{"params": model.parameters()}]

    param_dict: Dict[str, nn.Parameter] = dict(model.named_parameters())
    assigned: set = set()
    groups: List[Dict[str, Any]] = []

    for g in groups_cfg:
        group_name: str = g["name"]
        lr_mult: float = float(g.get("lr_mult", 1.0))
        params: List[nn.Parameter] = []
        for n, p in param_dict.items():
            if group_name in n and n not in assigned:
                params.append(p)
                assigned.add(n)
        if params:
            groups.append({"params": params, "lr": base_lr * lr_mult})

    # remaining params
    remaining: List[nn.Parameter] = [
        p for n, p in param_dict.items() if n not in assigned
    ]
    if remaining:
        groups.insert(0, {"params": remaining, "lr": base_lr})

    return groups
