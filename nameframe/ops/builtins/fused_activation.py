"""Built-in accelerated functions registered at import time."""

import torch
import torch.nn.functional as F

from nameframe.ops.decorators import accelerated


@accelerated("fused_gelu_dropout")
def fused_gelu_dropout(x: "torch.Tensor", p: float = 0.5, training: bool = True) -> "torch.Tensor":
    """Fused GELU activation followed by dropout.

    Args:
        x: Input tensor of any shape.
        p: Dropout probability.
        training: Whether in training mode (dropout active).

    Returns:
        Tensor with GELU activation + dropout applied.
    """
    import torch
    out: torch.Tensor = F.gelu(x)
    if training:
        out = F.dropout(out, p=p, training=True)
    return out


@accelerated("fused_layernorm_gelu")
def fused_layernorm_gelu(
    x: "torch.Tensor",
    normalized_shape: "tuple[int, ...]" = (64,),
    eps: float = 1e-5,
) -> "torch.Tensor":
    """Fused LayerNorm + GELU activation in a single path.

    Args:
        x: Input tensor.
        normalized_shape: LayerNorm normalized shape.
        eps: LayerNorm epsilon.

    Returns:
        Tensor after LayerNorm + GELU.
    """
    import torch
    import torch.nn as nn

    ln: nn.LayerNorm = nn.LayerNorm(normalized_shape, eps=eps).to(x.device)
    return F.gelu(ln(x))


@accelerated("fused_scale_mask_softmax")
def fused_scale_mask_softmax(
    x: "torch.Tensor",
    mask: "torch.Tensor",
    scale: float = 0.125,
) -> "torch.Tensor":
    """Fused Scale + Mask + Softmax for attention mechanisms.

    Args:
        x: Attention logits ``(B, H, N, N)``.
        mask: Mask tensor (broadcastable); 1 = keep, 0 = mask out.
        scale: Scaling factor (default ``1/sqrt(d_k)`` = 0.125 for d_k=64).

    Returns:
        Softmax probabilities.
    """
    x = x * scale
    x = x.masked_fill(mask == 0, float("-inf"))
    return F.softmax(x, dim=-1)
