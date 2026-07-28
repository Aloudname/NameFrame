# NameFrame

A PyTorch-based deep learning framework with swappable components.

## Getting Started

```python
from nameframe.registry import model

@model.register("my_model")
class MyModel:
    ...
```

## API Reference

- [Base API](api/base.md) — registry, protocols, seed & env
- [Config API](api/config.md) — configuration system
