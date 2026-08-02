"""
losses inherit pytorch `nn.Modules` with extra declaration of `config_schema`.

usage::

    import torch.nn as nn
    from nameframe.registry import loss

    @loss.register('my_cross_entropy')
    class MyCrossEntropy(nn.Module):
        config_schema = {
            "name": {"type": str, "help": "registry key."},
            "label_smoothing": {"type": float, "default": 0.0},
        }

        def __init__(self, label_smoothing=0.0):
            super().__init__()
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        def forward(self, pred, batch):
            return self.loss_fn(pred, batch.target)
"""

from nameframe.utils.typing import LossProtocol

__all__ = ["LossProtocol"]
