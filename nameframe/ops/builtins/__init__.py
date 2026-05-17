"""Built-in accelerated ops — import-time registration."""

from nameframe.ops.builtins.fused_activation import (
    fused_gelu_dropout,
    fused_layernorm_gelu,
    fused_scale_mask_softmax,
)

__all__ = [
    "fused_gelu_dropout",
    "fused_layernorm_gelu",
    "fused_scale_mask_softmax",
]
