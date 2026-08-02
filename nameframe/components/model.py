"""
models inherit pytorch `nn.Modules` with extra declaration of `config_schema`.

usage::

    import torch.nn as nn
    from nameframe.registry import model

    @model.register('my_vit')
    class MyViT(nn.Module):
        config_schema = {
            "name": {"type": str, "help": "registry key."},
            "num_layers": {"type": int, "default": 12},
            "hidden_dim": {"type": int, "default": 768},
        }

        def __init__(self, num_layers=12, hidden_dim=768):
            super().__init__()
            ...

        def forward(self, batch):
            return self.encoder(batch.data)
"""

from nameframe.utils.typing import ModelProtocol

__all__ = ["ModelProtocol"]
