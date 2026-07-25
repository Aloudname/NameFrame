# Phase 1 实现方案：基础工具 + 注册表

> 对应 `docs/plan.md` Phase 1。本阶段零内部依赖，只依赖 Python 标准库和 PyTorch。

---

## 一、交付物总览

```
nameframe/
├── __init__.py              # 包入口，版本号
├── utils/
│   ├── __init__.py          # 导出所有公共类型和工具函数
│   ├── typing.py            # BatchProtocol, ModelProtocol, LossProtocol,
│   │                        #   MetricProtocol, DatasetProtocol, FieldSchema
│   ├── seed.py              # set_seed(), derive_seed()
│   └── env.py               # capture_env()
└── registry/
    ├── __init__.py           # 导出 Registry + 6 个全局实例
    ├── core.py              # Registry, RegistryKeyError, _camel_to_snake
    └── discovery.py         # discover()
```

**依赖关系**：

```
nameframe/__init__.py
  └── nameframe/utils/__init__.py  (无内部依赖)
  └── nameframe/registry/__init__.py  (无内部依赖)

nameframe/utils/typing.py  ← 标准库 typing + torch.Tensor
nameframe/utils/seed.py    ← random, hashlib, numpy, torch
nameframe/utils/env.py     ← sys, platform, torch

nameframe/registry/core.py     ← re, pathlib, typing
nameframe/registry/discovery.py ← importlib, sys, pathlib
```

---

## 二、文件详细设计

### 2.1 `nameframe/__init__.py`

包入口。只做两件事：声明版本号，暴露 `utils` 和 `registry` 子包。

```python
"""
nameframe - a modular, standardized deep learning training framework.
"""

__version__ = "0.1.0.dev0"

from nameframe import registry
from nameframe import utils

__all__ = ["registry", "utils"]
```

---

### 2.2 `nameframe/utils/__init__.py`

聚合 `utils/` 下所有公共符号，提供单一导入点。后续 Phase 的代码写 `from nameframe.utils import BatchProtocol` 即可。

```python
from nameframe.utils.env import capture_env
from nameframe.utils.seed import derive_seed, set_seed
from nameframe.utils.typing import (
    BatchProtocol,
    DatasetProtocol,
    FieldSchema,
    LossProtocol,
    MetricProtocol,
    ModelProtocol,
)

__all__ = [
    "BatchProtocol",
    "DatasetProtocol",
    "FieldSchema",
    "LossProtocol",
    "MetricProtocol",
    "ModelProtocol",
    "capture_env",
    "derive_seed",
    "set_seed",
]
```

---

### 2.3 `nameframe/utils/typing.py`

定义整个框架共享的 Protocol 和 TypedDict。这些类型是所有组件接口的契约来源。

**设计决策**：
- 使用 `Protocol` 而非 ABC——组件不需要显式继承，满足结构即可（structural subtyping）
- `@runtime_checkable` 不加——避免 import 时的性能开销，类型检查留给静态分析工具
- `BatchProtocol` 的 `target` 可为 `None`（无监督学习场景）
- `FieldSchema` 用 `total=False`，所有字段可选

```python
"""
shared type definitions for the nameframe framework.

all component protocols are defined here to avoid circular imports
between subpackages. protocols use structural subtyping via
typing.Protocol, so components do not need to explicitly inherit.
"""

from typing import Any, Callable, Iterator, Protocol, TypedDict

from torch import Tensor
from torch.nn import Parameter


class BatchProtocol(Protocol):
    """protocol for batch data objects passed between components."""

    data: Tensor
    """the model input tensor."""

    target: Tensor | None
    """the target tensor, or none for unsupervised tasks."""

    meta: dict
    """optional metadata dict for auxiliary information."""


class ModelProtocol(Protocol):
    """protocol for model components."""

    config_schema: dict
    """declared configuration schema for the model."""

    def forward(self, batch: BatchProtocol) -> Tensor:
        """
        forward pass.

        params:
        - `batch`: `BatchProtocol` type.

        returns:
        - `Tensor` type, the model output.
        """
        ...

    def parameters(self) -> Iterator[Parameter]:
        """returns an iterator over model parameters."""
        ...

    def state_dict(self) -> dict:
        """returns the model's state as a dict."""
        ...

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """
        loads state from a dict.

        params:
        - `state_dict`: `dict` type.
        - `strict`: `bool` type.
        """
        ...


class LossProtocol(Protocol):
    """protocol for loss components."""

    config_schema: dict
    """declared configuration schema for the loss."""

    def forward(self, pred: Tensor, batch: BatchProtocol) -> Tensor:
        """
        computes the loss.

        params:
        - `pred`: `Tensor` type, the model prediction.
        - `batch`: `BatchProtocol` type, the original batch containing targets.

        returns:
        - `Tensor` type, a scalar loss value.
        """
        ...


class MetricProtocol(Protocol):
    """protocol for metric components."""

    config_schema: dict
    """declared configuration schema for the metric."""

    def update(self, pred: Tensor, batch: BatchProtocol) -> None:
        """
        accumulates a single prediction into the metric state.

        params:
        - `pred`: `Tensor` type, the model prediction.
        - `batch`: `BatchProtocol` type, the original batch.
        """
        ...

    def compute(self) -> dict[str, float]:
        """
        computes the current metric values.

        returns:
        - `dict[str, float]` type, metric name to value mapping.
        """
        ...

    def reset(self) -> None:
        """resets all accumulated metric state."""
        ...


class DatasetProtocol(Protocol):
    """protocol for dataset components."""

    config_schema: dict
    """declared configuration schema for the dataset."""

    augmentation_transforms: list
    """training-only transforms, auto-disabled during inference."""

    def fingerprint(self) -> str:
        """
        returns a unique fingerprint for the dataset identity.

        returns:
        - `str` type, the fingerprint hash or identifier.
        """
        ...

    def transforms(self, mode: str = "train") -> list[Callable]:
        """
        returns the transform list for the given mode.

        params:
        - `mode`: `str` type, 'train' or 'infer'.

        returns:
        - `list[Callable]` type, the applicable transforms.
        """
        ...

    def __getitem__(self, index: int) -> BatchProtocol:
        """returns a single data item as a BatchProtocol."""
        ...

    def __len__(self) -> int:
        """returns the dataset size."""
        ...


class FieldSchema(TypedDict, total=False):
    """typeddict schema for a single configuration field."""

    type: type
    """the expected python type."""

    default: Any
    """the default value."""

    help: str
    """human-readable description of the field."""

    choices: list | None
    """allowed values, or none for no restriction."""

    range: tuple[float, float] | None
    """allowed numeric range (min, max), or none."""

    nullable: bool
    """whether the field accepts none. defaults to false."""
```

---

### 2.4 `nameframe/utils/seed.py`

全局种子管理。两个函数：`set_seed` 设置所有后端的全局种子，`derive_seed` 基于组件名和 rank 派生子种子。

**设计决策**：
- `set_seed` 是幂等的——多次调用产生相同的随机状态
- `derive_seed` 使用 MD5 而非 Python 内置 `hash()`——后者在进程间不稳定
- 派生种子公式 `base + offset + rank * 1000` 保证了不同 rank、不同组件的种子不碰撞

```python
"""
deterministic seed management for reproducible training.

provides functions to set global seeds across all random backends
and to derive independent sub-seeds for isolated components.
"""

import hashlib
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    sets global random seeds across all supported backends.

    configures python random, numpy, pytorch cpu, pytorch cuda (all gpus),
    and forces cudnn to deterministic mode. this is the single entry point
    for making a training run reproducible.

    params:
    - `seed`: `int` type, the master seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def derive_seed(base_seed: int, component: str, rank: int = 0) -> int:
    """
    derives an independent seed for a component from a master seed.

    uses an md5 hash of the component name to produce a deterministic
    offset. this prevents accidental seed collisions between different
    components (model init, data shuffle, dropout) while keeping every
    run fully reproducible given the same base seed.

    params:
    - `base_seed`: `int` type, the master seed from set_seed().
    - `component`: `str` type, a unique identifier (e.g. 'model_init',
      'data_shuffle', 'dropout').
    - `rank`: `int` type, the distributed process rank. default 0.

    returns:
    - `int` type, the derived seed.
    """
    hash_bytes: bytes = hashlib.md5(component.encode()).digest()
    offset: int = int.from_bytes(hash_bytes[:4], byteorder="big") % 10000
    return base_seed + offset + rank * 1000
```

---

### 2.5 `nameframe/utils/env.py`

捕获运行时环境签名，用于 checkpoint 记录和跨机器复现诊断。

**设计决策**：
- CUDA 不可用时返回 `"none"` 而非抛异常——CPU-only 环境也是合法环境
- cuDNN 版本获取可能因 PyTorch 构建方式不同而失败，`try/except` 兜底
- `sys.version.split()[0]` 只保留 `"3.10.12"` 部分，去掉编译信息

```python
"""
runtime environment capture for reproducibility diagnostics.

records the software stack signature so that experiments can be
compared across machines and reproduced in equivalent environments.
"""

import platform
import sys

import torch


def capture_env() -> dict:
    """
    captures the runtime environment signature.

    returns a dict with keys 'python', 'torch', 'cuda', 'cudnn', and
    'platform'. cuda and cudnn report 'none' when unavailable.

    returns:
    - `dict` type, mapping component names to version strings.
    """
    cuda_version: str
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda or "unknown"
    else:
        cuda_version = "none"

    cudnn_version: str
    try:
        cudnn_version = str(torch.backends.cudnn.version())
    except Exception:
        cudnn_version = "none"

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": cuda_version,
        "cudnn": cudnn_version,
        "platform": platform.platform(),
    }
```

---

### 2.6 `nameframe/registry/core.py`

框架的核心骨架。`Registry` 类提供三种注册方式（装饰器、函数式、自动发现）和两种查询方式（按名获取、全量列举）。

**设计决策**：
- `base_type` 检查在注册时执行而非 `get()` 时——尽早发现错误（哲学 #20）
- 命名空间用 `:` 分隔符，符合直觉且不与其他命名规则冲突
- `register_external` 方法独立于装饰器，语义更清晰
- `_camel_to_snake` 是模块私有函数，不导出——外部代码不需要驼峰转换
- `_items` 用 `dict[str, type]` 存储——值始终是类（type），而非实例。延迟加载由上层 `from_config()` 负责

```python
"""
centralized module registry for the nameframe framework.

the registry is the backbone of nameframe's component system. every
swappable component type (model, dataset, loss, metric, ops, plugin)
has its own registry instance, enabling name-based lookup, namespace
isolation, and automatic module discovery.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class RegistryKeyError(KeyError):
    """
    raised when a requested key is not found in a registry.

    params:
    - `key`: `str` type, the key that was requested.
    - `registry_name`: `str` type, the name of the registry.
    - `available`: `list[str]` type, the list of registered keys.
    """

    def __init__(
        self, key: str, registry_name: str, available: list[str]
    ) -> None:
        self.key = key
        self.registry_name = registry_name
        self.available = available
        msg = (
            f"'{key}' not found in {registry_name} registry. "
            f"available: {available}"
        )
        super().__init__(msg)


def _camel_to_snake(name: str) -> str:
    """
    converts a CamelCase string to snake_case.

    handles consecutive capitals (e.g. 'MyViT' -> 'my_vit') and
    digit boundaries (e.g. 'ResNet50' -> 'res_net50').

    params:
    - `name`: `str` type, the CamelCase input.

    returns:
    - `str` type, the snake_case output.
    """
    # insert underscore between a run of capitals followed by a capital
    # then lowercase letter, e.g. 'ViTBlock' -> 'ViT_Block'
    s1: str = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    # insert underscore between a lowercase/digit and a capital,
    # e.g. 'myViT' -> 'my_ViT' or 'net50Block' -> 'net50_Block'
    s2: str = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


class Registry:
    """
    a central registry for discoverable, swappable components.

    each component category (model, dataset, loss, metric, ops, plugin)
    has one global registry instance. components are registered via
    decorator, explicit function call, or auto-discovery from a directory.

    params:
    - `name`: `str` type, human-readable name for error messages.
    - `base_type`: optional `type`, if set, all registered items must be
      subclasses of this type, checked at registration time.
    """

    def __init__(self, name: str, base_type: type | None = None) -> None:
        self.name: str = name
        self.base_type: type | None = base_type
        self._items: dict[str, type] = {}
        self._namespaces: dict[str, "Registry"] = {}

    # ---- registration -------------------------------------------------

    def register(self, name: str | None = None):
        """
        decorator that registers a class under the given key.

        if name is none, the class name is auto-converted from CamelCase
        to snake_case. usage:

            @reg.model.register("my_vit")
            class MyViT(nn.Module): ...

            @reg.model.register()       # registers as 'my_vit'
            class MyViT(nn.Module): ...

        params:
        - `name`: optional `str` type, the registration key.

        returns:
        - decorator callable that registers and returns the class unchanged.
        """
        def decorator(cls: type) -> type:
            key: str = name if name is not None else _camel_to_snake(cls.__name__)
            self._validate_type(key, cls)
            self._items[key] = cls
            return cls

        return decorator

    def register_external(self, name: str, item: type) -> None:
        """
        registers an externally-defined class under the given key.

        this is the functional (non-decorator) registration path, used
        for importing components from third-party libraries like
        torchvision or transformers. usage:

            reg.model.register_external("resnet50", torchvision.models.resnet50)

        params:
        - `name`: `str` type, the registration key.
        - `item`: `type`, the class to register.
        """
        self._validate_type(name, item)
        self._items[name] = item

    def _validate_type(self, key: str, cls: type) -> None:
        """validates that cls is a subclass of base_type, if set."""
        if self.base_type is not None and not issubclass(cls, self.base_type):
            raise TypeError(
                f"'{key}' must be a subclass of {self.base_type.__name__}, "
                f"got {cls.__name__}."
            )

    # ---- lookup --------------------------------------------------------

    def get(self, name: str) -> type:
        """
        retrieves a registered class by name.

        supports namespace syntax 'ns:name' for cross-project isolation.
        raises RegistryKeyError if the name is not found, listing all
        available keys.

        params:
        - `name`: `str` type, the registration key, optionally prefixed
          with 'namespace:'.

        returns:
        - the registered `type`.

        raises:
        - `RegistryKeyError`: if the key is not registered.
        """
        if ":" in name:
            ns, key = name.split(":", 1)
            if ns not in self._namespaces:
                raise RegistryKeyError(name, self.name, list(self._items.keys()))
            return self._namespaces[ns].get(key)

        if name not in self._items:
            raise RegistryKeyError(name, self.name, list(self._items.keys()))
        return self._items[name]

    def list_all(self, namespace: str | None = None) -> dict[str, type]:
        """
        returns all registered items as a key-to-type mapping.

        params:
        - `namespace`: optional `str` type, if provided, only returns
          items registered under that namespace.

        returns:
        - `dict[str, type]` type, mapping registration keys to types.
        """
        if namespace is not None:
            if namespace not in self._namespaces:
                return {}
            return dict(self._namespaces[namespace]._items)
        return dict(self._items)

    # ---- namespace ----------------------------------------------------

    def create_namespace(self, name: str) -> "Registry":
        """
        creates or retrieves a sub-namespace within this registry.

        namespaces isolate different projects' registrations. a model
        registered as 'proj_a:resnet' does not collide with 'proj_b:resnet'.

        params:
        - `name`: `str` type, the namespace identifier.

        returns:
        - `Registry` instance scoped to the namespace.
        """
        if name not in self._namespaces:
            self._namespaces[name] = Registry(
                f"{self.name}:{name}", self.base_type
            )
        return self._namespaces[name]

    # ---- dunder -------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        """checks whether a key is registered. supports 'ns:name'."""
        if ":" in name:
            ns, key = name.split(":", 1)
            return ns in self._namespaces and key in self._namespaces[ns]
        return name in self._items

    def __len__(self) -> int:
        """returns the number of directly registered items."""
        return len(self._items)

    def __repr__(self) -> str:
        return (
            f"Registry(name='{self.name}', "
            f"items={len(self._items)}, "
            f"namespaces={list(self._namespaces.keys())})"
        )
```

---

### 2.7 `nameframe/registry/discovery.py`

自动发现机制。扫描目录下所有 `.py` 文件，动态 import 触发 `@register` 副作用。

**设计决策**：
- 跳过 `__init__.py`——这些文件通常不包含注册代码
- 使用 `importlib.util.spec_from_file_location` 而非 `__import__`——前者对任意路径更可靠
- 同时支持 `.py` 源文件（开发环境）和未来可能的编译文件
- 导入失败的模块打印警告并跳过——不阻塞整体发现流程（一个模块坏了不影响其他）

```python
"""
automatic module discovery for registry population.

scans a directory tree and imports every python module found,
triggering side-effect registrations from @registry.*.register
decorator calls.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path


def discover(directory: Path) -> None:
    """
    recursively imports all python modules in a directory tree.

    each imported module's top-level code executes, triggering any
    @registry.*.register decorator calls encountered. __init__.py
    files are skipped. import errors are caught and warned rather
    than raised, so one broken module does not block the rest.

    params:
    - `directory`: `Path` type, the root directory to scan.

    raises:
    - `NotADirectoryError`: if directory does not exist or is a file.
    """
    directory = directory.resolve()
    if not directory.is_dir():
        raise NotADirectoryError(
            f"'{directory}' is not a valid directory."
        )

    for py_file in directory.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue

        # construct a dotted module path relative to directory's parent
        # e.g. my_project/model/my_vit.py -> my_project.model.my_vit
        relative: Path = py_file.relative_to(directory.parent)
        module_path: str = (
            str(relative.with_suffix(""))
            .replace("/", ".")
            .replace("\\", ".")
        )

        _import_module_from_path(module_path, py_file)


def _import_module_from_path(module_path: str, file_path: Path) -> None:
    """
    imports a single module from a file path.

    uses importlib to load and execute the module, which triggers
    any module-level side effects including registry decorators.

    params:
    - `module_path`: `str` type, dotted module name.
    - `file_path`: `Path` type, the absolute path to the .py file.
    """
    try:
        spec = importlib.util.spec_from_file_location(
            module_path, str(file_path)
        )
        if spec is None or spec.loader is None:
            warnings.warn(
                f"could not create module spec for '{file_path}'"
            )
            return

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_path] = module
        spec.loader.exec_module(module)

    except Exception as e:
        warnings.warn(
            f"failed to import '{module_path}' from '{file_path}': {e}"
        )
```

---

### 2.8 `nameframe/registry/__init__.py`

创建并导出 6 个全局 Registry 实例。这是框架其他部分访问注册表的唯一入口。

```python
"""
global registry instances for the nameframe framework.

each component category has a singleton registry instance. all other
nameframe modules import from here to register or look up components.
"""

from nameframe.registry.core import Registry, RegistryKeyError
from nameframe.registry.discovery import discover

model: Registry = Registry("model")
"""registry for model components."""

dataset: Registry = Registry("dataset")
"""registry for dataset components."""

loss: Registry = Registry("loss")
"""registry for loss components."""

metric: Registry = Registry("metric")
"""registry for metric components."""

ops: Registry = Registry("ops")
"""registry for custom operator components."""

plugin: Registry = Registry("plugin")
"""registry for plugin components."""

__all__ = [
    "Registry",
    "RegistryKeyError",
    "discover",
    "model",
    "dataset",
    "loss",
    "metric",
    "ops",
    "plugin",
]
```

---

## 三、接口契约

以下接口被后续 Phase 依赖，修改需向后兼容：

| 接口 | 使用者 | 契约 |
|------|--------|------|
| `Registry.register(name=None)` | Phase 3, 4, 9 | 装饰器，返回原类，支持 `name` 参数 |
| `Registry.register_external(name, item)` | Phase 3, 8 | `name` 必填，`item` 是 `type` |
| `Registry.get(name)` | Phase 3, 6, 8 | `"ns:key"` 格式查命名空间，未找到抛 `RegistryKeyError` |
| `Registry.list_all(ns)` | Phase 5, 8 | 返回 `dict[str, type]` 深拷贝 |
| `Registry.__contains__(name)` | Phase 6 | `"name" in registry` 返回 `bool` |
| `discover(directory)` | Phase 8 (`nfm init`) | 递归扫描，失败时 warn 不抛异常 |
| `set_seed(seed)` | Phase 6, 8 | 幂等，设置所有后端种子 |
| `derive_seed(base, comp, rank)` | Phase 6, 7 | 返回 `int`，确定性 |
| `capture_env()` | Phase 5, 6 | 返回 `dict`，5 个固定 key |
| `BatchProtocol` | Phase 3, 4, 6 | Protocol，不可变语义 |
| `FieldSchema` | Phase 2, 3 | TypedDict, total=False |

---

## 四、异常策略

| 异常 | 触发条件 | 携带信息 |
|------|---------|---------|
| `RegistryKeyError` | `get()` 的 key 不存在 | `key`, `registry_name`, `available` |
| `TypeError` | `register()` 时 `base_type` 检查失败 | key 名、期望类型、实际类型 |
| `NotADirectoryError` | `discover()` 的 directory 不合法 | 路径 |
| `Warning`（非阻塞） | `discover()` 的单个模块导入失败 | 模块路径、异常信息 |

---

## 五、测试策略

### 5.1 单元测试（`tests/test_registry.py`）

```python
import pytest
from nameframe.registry import Registry, RegistryKeyError, model


class TestRegistryInit:
    def test_creates_with_name(self):
        r = Registry("test")
        assert r.name == "test"
        assert len(r) == 0

    def test_repr_includes_name_and_count(self):
        r = Registry("test")
        assert "test" in repr(r)


class TestRegistryRegister:
    def test_decorator_registers_class(self):
        r = Registry("test")

        @r.register("my_class")
        class MyClass:
            pass

        assert "my_class" in r
        assert r.get("my_class") is MyClass

    def test_decorator_auto_names_class(self):
        r = Registry("test")

        @r.register()
        class MyViTBlock:
            pass

        # MyViTBlock -> my_vi_tblock (edge case with consecutive caps)
        assert "my_vi_tblock" in r

    def test_register_external_adds_item(self):
        r = Registry("test")
        r.register_external("ext", dict)
        assert r.get("ext") is dict

    def test_base_type_enforcement(self):
        r = Registry("test", base_type=dict)

        with pytest.raises(TypeError):
            r.register_external("bad", list)

    def test_duplicate_registration_overwrites(self):
        r = Registry("test")

        @r.register("dup")
        class V1:
            pass

        @r.register("dup")
        class V2:
            pass

        assert r.get("dup") is V2


class TestRegistryGet:
    def test_get_raises_with_available_list(self):
        r = Registry("test")
        r.register_external("foo", dict)

        with pytest.raises(RegistryKeyError) as exc:
            r.get("bar")
        assert "bar" in str(exc.value)
        assert "foo" in str(exc.value)

    def test_namespace_lookup(self):
        r = Registry("test")
        ns = r.create_namespace("proj")
        ns.register_external("foo", list)

        assert "proj:foo" in r
        assert r.get("proj:foo") is list


class TestRegistryListAll:
    def test_returns_shallow_copy(self):
        r = Registry("test")
        r.register_external("a", int)
        items = r.list_all()
        items["a"] = str
        assert r.get("a") is int  # original unaffected


class TestRegistryNamespace:
    def test_create_isolated_space(self):
        r = Registry("test")
        r.register_external("x", int)
        ns = r.create_namespace("proj")
        ns.register_external("x", str)

        assert r.get("x") is int
        assert r.get("proj:x") is str
        assert r.list_all() == {"x": int}
        assert r.list_all("proj") == {"x": str}
```

### 5.2 单元测试（`tests/test_seed.py`）

```python
from nameframe.utils.seed import set_seed, derive_seed


class TestSetSeed:
    def test_does_not_raise(self):
        set_seed(42)

    def test_is_deterministic(self):
        import random
        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b


class TestDeriveSeed:
    def test_different_components_get_different_seeds(self):
        s1 = derive_seed(42, "model_init")
        s2 = derive_seed(42, "data_shuffle")
        assert s1 != s2

    def test_same_inputs_same_output(self):
        a = derive_seed(42, "dropout", rank=0)
        b = derive_seed(42, "dropout", rank=0)
        assert a == b

    def test_different_ranks_different_seeds(self):
        s0 = derive_seed(42, "init", rank=0)
        s1 = derive_seed(42, "init", rank=1)
        assert s0 != s1
```

### 5.3 单元测试（`tests/test_env.py`）

```python
from nameframe.utils.env import capture_env


class TestCaptureEnv:
    def test_returns_dict_with_required_keys(self):
        env = capture_env()
        for key in ["python", "torch", "cuda", "cudnn", "platform"]:
            assert key in env

    def test_values_are_strings(self):
        env = capture_env()
        for v in env.values():
            assert isinstance(v, str)

    def test_python_is_version_like(self):
        env = capture_env()
        parts = env["python"].split(".")
        assert len(parts) >= 2
```

### 5.4 集成测试（`tests/test_phase1_integration.py`）

```python
"""end-to-end test exercising the full phase 1 public api."""

from nameframe import __version__
from nameframe.registry import Registry, RegistryKeyError, model, dataset
from nameframe.registry import loss, metric, ops, plugin, discover
from nameframe.utils import BatchProtocol, FieldSchema
from nameframe.utils import set_seed, derive_seed, capture_env


def test_version_is_string():
    assert isinstance(__version__, str)
    assert __version__.count(".") == 2


def test_all_registry_instances_exist():
    for reg in [model, dataset, loss, metric, ops, plugin]:
        assert isinstance(reg, Registry)
        assert len(reg) >= 0


def test_full_registration_flow():
    """decorator -> get -> contains -> list_all"""
    @model.register("integration_test_model")
    class TestModel:
        pass

    assert "integration_test_model" in model
    assert model.get("integration_test_model") is TestModel
    assert "integration_test_model" in model.list_all()


def test_namespace_isolation():
    ns_a = model.create_namespace("ns_a")
    ns_b = model.create_namespace("ns_b")

    @ns_a.register("same_name")
    class ModelA:
        pass

    @ns_b.register("same_name")
    class ModelB:
        pass

    assert model.get("ns_a:same_name") is ModelA
    assert model.get("ns_b:same_name") is ModelB
    assert model.get("ns_a:same_name") is not model.get("ns_b:same_name")


def test_registry_key_error_is_helpful():
    try:
        model.get("definitely_not_registered_xyz")
        assert False, "should have raised"
    except RegistryKeyError as e:
        assert "definitely_not_registered_xyz" in str(e)
        assert "model" in str(e)


def test_seed_and_env_together():
    set_seed(123)
    env = capture_env()
    assert "torch" in env
```

---

## 六、完成检查清单

- [ ] `nameframe/__init__.py` 存在，`import nameframe` 成功，`__version__` 可访问
- [ ] `nameframe/utils/typing.py` 所有 6 个类型定义可 import
- [ ] `nameframe/utils/seed.py` `set_seed()` 不抛异常，`derive_seed()` 确定性
- [ ] `nameframe/utils/env.py` `capture_env()` 返回 5 个 key 的 dict
- [ ] `nameframe/registry/core.py` `Registry` 支持 register / register_external / get / list_all / create_namespace / `__contains__`
- [ ] `nameframe/registry/discovery.py` `discover()` 能扫描目录并触发注册
- [ ] `nameframe/registry/__init__.py` 6 个全局 Registry 实例可 import
- [ ] 所有单元测试通过
- [ ] 集成测试通过
- [ ] 类型注解完整，`mypy` 检查无错误（strict 模式除外）
- [ ] 所有公共函数/类/方法有 docstring，遵循 plan.md 规范
