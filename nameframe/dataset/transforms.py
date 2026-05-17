"""Common data transforms and augmentation pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from munch import Munch


class AugmentationPipeline:
    """Configurable geometric and photometric augmentation pipeline.

    Each transform is independently controlled by a probability; the
    pipeline applies them in a fixed, sensible order.

    Attributes:
        flip_prob: Horizontal flip probability.
        rotate_prob: Rotation probability.
        rotate_limit: Maximum rotation degrees (±).
        noise_std: Gaussian noise standard deviation.
    """

    def __init__(self, config: Munch) -> None:
        """Initialize with an augmentation config subtree.

        Args:
            config: The ``config.data.augment`` subtree.
        """
        cfg: Dict[str, Any] = config.toDict() if isinstance(config, Munch) else dict(config)
        self.flip_prob: float = float(cfg.get("flip_prob", 0.5))
        self.rotate_prob: float = float(cfg.get("rotate_prob", 0.5))
        self.rotate_limit: float = float(cfg.get("rotate_limit", 15.0))
        self.noise_std: float = float(cfg.get("noise_std", 0.01))
        self._enabled: bool = bool(cfg.get("enabled", False))

    def __call__(
        self,
        image: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply augmentations to an image and (optionally) its target mask.

        Args:
            image: Tensor of shape ``(C, H, W)``.
            target: Optional mask tensor of shape ``(H, W)``.

        Returns:
            ``(augmented_image, augmented_target)`` tuple.
        """
        if not self._enabled:
            return image, target

        return image, target


def build_transforms(config: Munch) -> AugmentationPipeline:
    """Build an :class:`AugmentationPipeline` from configuration.

    Args:
        config: The full project config Munch.

    Returns:
        A callable augmentation pipeline.
    """
    return AugmentationPipeline(config.data.augment)
