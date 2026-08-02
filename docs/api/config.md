# 配置系统 API

> `nameframe/config` 管理所有配置项。该部分内部依赖（`FieldSchema`，外部仅依赖 *Python 3.10+* 标准库和 PyYAML。

---

## 一、全局入口

配置系统在顶层的入口是 `nameframe.config`，包含四个比较重要的字段：

| 符号 | 类型 | 说明 |
|------|------|------|
| `Config` | 类 | 配置树 |
| `validate_schema` | 函数 | 配置校验 |
| `ConfigConflictError` | 异常 | 校验失败 |
| `FrozenConfigError` | 异常 | 修改警告 |

```python
from nameframe.config import (
    Config,
    ConfigConflictError,
    FrozenConfigError,
    validate_schema,
)
```

---

## 二、`Config` 配置树

从用户的角度看，无论是超参数、模型结构参数还是各种杂项参数，这些不同类别的参数应该由统一的入口管理，也就是 `.yaml` 配置文件。最终，这些参数会散开到各个组件中。为描述这一行为，配置系统被抽象成了一个树型结构 `Config`，它是配置管线的 **主要** 入口和唯一出口。

### 2.1 配置入口

#### 2.1.1 文件入口 `.yaml`

用户在 `.yaml` 文件中可配置两种格式的参数。大多情况，参数是确定的，直接用键值对配置；当需要围绕某一范围动态生成参数时，可用占位符 `${search: ...}` 赋值。另外，还支持使用 `!include` 指令，通过相对路径引用其他 `.yaml` 文件。例如：

```yaml
# exp.yaml

# 引用
!include model/resnet50.yaml

model:
  name: resnet50
training:
  # 占位赋值
  lr: "${search: loguniform(1e-5, 1e-2)}"
  weight_decay: "${search: uniform(1e-5, 1e-2)}"
  epochs: 100
```

> **[BUG]** 由于 pyyaml 的节点类型判断在 tag 处理之前，`!include` 不可与 `<<:` YAML 合并键组合使用。

#### 2.1.2 CLI 入口

为方便调试，用户可通过命令行参数 **暂时** 遮盖 `.yaml` 文件中的配置。例如：

```bash
nfm train --model.num_layers 50 --training.lr "${search: loguniform(1e-5, 1e-2)}"
```

这种遮盖是一次性的，并不会修改原始 `.yaml` 文件。

### 2.2 `Config` 接口

`Config` 配置树提供一些接口，用于管理参数的加载、解析等流程。

#### 2.2.1 配置加载

首先调用 `Config.from_yaml()` 方法，将来自 `.yaml` 和 CLI 的配置合并，加载为 `Config` 对象。该方法不需提前实例化 `Config` 便可直接调用：

```python
@classmethod
def from_yaml(cls, path: Path, overrides: dict | None = None) -> Config
```

`overrides` 参数用于在加载配置时遮盖 `.yaml` 文件中的值。这一过程由 CLI 参数触发并自动传入，不需手动操作。例如：

```python
from pathlib import Path
from nameframe.config import Config

config = Config.from_yaml(Path("exp.yaml"))

# CLI 遮盖
config = Config.from_yaml(
    Path("exp.yaml"),
    overrides={"training": {"lr": 5e-4}},
)
```

#### 2.2.2 配置解析

对配置树实例中的 `${search: ...}` 占位符，需调用实例的 `resolve()` 方法将其解析为可进行参数搜索的对象。该方法对配置树实例进行 **原地修改**，将占位符替换为一个方便搜索的 `SearchSpace` 实例。

```python
def resolve(self) -> None
```

例如：

```python
# exp.yaml
# training:
#   lr: "${search: loguniform(1e-5, 1e-2)}"

# usage
cfg = Config.from_yaml(Path("exp.yaml"))
cfg.resolve()
type(cfg.training.lr)  # -> <class 'SearchSpace'>
```

#### 2.2.3 配置冻结

框架不允许一切在运行时修改配置树的行为。解析完毕后，调用实例的 `freeze()` 方法冻结整棵配置树，将其变为只读对象。冻结后任何写入尝试都会抛出 `FrozenConfigError`。
 
```python
def freeze(self) -> None
```

框架的编排层通过 `nfm train` 封装了配置加载、解析、冻结这一管线，用户一般无需实现该流程。若需自己编排配置流程，三者必须 **严格按顺序调用**。例如：

```python
from pathlib import Path
from nameframe.config import Config

cfg = Config.from_yaml(Path("exp.yaml"), overrides={"training": {"lr": 5e-4}})
cfg.resolve()
cfg.freeze()

cfg.model.name = "vit"  # FrozenConfigError
```

#### 2.2.4 属性访问

加载后的 `Config` 对象像普通 Python 对象一样通过 `.` 访问，嵌套字典会自动转为嵌套属性：

```python
config.model.name  # -> "resnet50"
config.training.optimizer.type  # -> "adam"
```

支持 `in` 关键字和 `len()`：

```python
"lr" in config.training  # -> True
len(config.training)  # -> 4，命名空间拥有字段数
```

#### 2.2.5 其他方法

`Config` 类提供一些内置方法或属性来管理配置的状态和操作。首先是 `is_frozen` 属性，用于检查配置树是否已被冻结：

```python
@property
def is_frozen(self) -> bool
```

```python
config = Config({"lr": 0.01})
config.is_frozen  # -> False
config.freeze()
config.is_frozen  # -> True
```

`get_namespace` 方法用于获取配置树中某个命名空间的内容，返回一个纯 `dict`：

```python
def get_namespace(self, ns: str) -> dict
```

该方法用于获取关键词参数字典，注入各组件的构造函数（`**kwargs`）。

```python
# exp.yaml:
# model:
#   backbone:
#     name: resnet50
#     pretrained: true

params = config.get_namespace("model.backbone")
# → {"name": "resnet50", "pretrained": True}
```

`to_dict` 方法可将配置树整个导出为 `dict`，用于 checkpoint 快照或序列化。修改返回的字典不会影响原始 `Config` 对象。

```python
def to_dict(self) -> dict
```

---

## 三、`validate_schema` 配置校验

配置文件被实例化为 `Config` 对象后，就要准备注入各种组件类，构造组件实例了。不过在此之前，需要检查某些参数配置是否合理。我们不希望传入的配置项不符合该种组件的规定，或者参数的值太离谱。[`api/base.md`](./base.md) 介绍了所有类型组件都有的属性 `FieldSchema`，它正是用于规范传入配置项的。将 `Config` 对象的参数与 `FieldSchema` 进行校验，确保在参数注入组件前拦截 **字段**、**类型** 和 **值域** 的错误。下面的方法封装了校验流程，失败抛出 `ConfigConflictError`。

```python
def validate_schema(params: dict, schema: dict[str, FieldSchema]) -> None
```

假设某模型定义了 `config_schema`：

```python
class MyModel(nn.Module):
    config_schema = {
        "hidden_dim": {
            "type": int,
            "default": 512,
            "range": (64, 2048),
            "help": "隐藏层维度",
        },
        "activation": {
            "type": str,
            "choices": ["relu", "gelu", "silu"],
        },
    }
```

校验流程：

```python
from nameframe.config import validate_schema
from nameframe.utils.typing import FieldSchema

schema: dict[str, FieldSchema] = {
    "hidden_dim": {"type": int, "default": 512, "range": (64, 2048)},
    "activation": {"type": str, "choices": ["relu", "gelu", "silu"]},
}

# 正常
validate_schema({"hidden_dim": 256, "activation": "gelu"}, schema)

# 字段错误
validate_schema({"hidden_dim": 256, "aCTivaWion": "gelu"}, schema)
# -> "unknown field 'aCTivaWion', did you mean 'activation'?"

# 值域、类型错误
validate_schema({"hidden_dim": "big", "activation": "tanh"}, schema)
# -> [
#     "field 'hidden_dim' expects int, got str (value='big').",
#     "field 'activation' has value 'tanh', but must be one of ['relu', 'gelu', 'silu'].",
# ]
```

---

## 四、`SearchSpace` 搜索空间

前面说到，`Config.resolve()` 方法将 `${search: ...}` 占位符解析为一个 `SearchSpace` 类型的实例。`SearchSpace` 是一个轻量的数据类型，供 CLI 编排层的超参搜索使用：

```python
@dataclass
class SearchSpace:
    expression: str  # ${search: ...} 内的原始表达式
```

方法尚未在 `config/__init__.py` 开放，但后续 CLI 编排层的 sweep 引擎会直接使用它。

---

## 五、应用例

我们用一个端到端示例，展示如何结合底层的注册表系统和配置系统。

首先定义配置文件。`exp.yaml` 负责实验配置，`model/vit_base.yaml` 负责模型配置，`loss/cross_entropy.yaml` 负责损失函数配置。

```yaml
# ./exp.yaml
model: !include model/vit_base.yaml
loss: !include loss/cross_entropy.yaml
training:
  lr: "${search: loguniform(1e-5, 1e-2)}"
  epochs: 100
  optimizer:
    type: adam
    weight_decay: 1e-4
```

```yaml
# ./model/vit_base.yaml
name: vit_base
image_size: 224
patch_size: 16
dim: 768
depth: 12
heads: 12
dropout: 0.1
```

```yaml
# ./loss/cross_entropy.yaml
name: cross_entropy
weight: 1.0
```

然后定义 Python 端：

```python
from pathlib import Path
from nameframe.config import Config, validate_schema
from nameframe.registry import model, loss
import torch.nn as nn

# 加载、校验配置
config = Config.from_yaml(
    Path("exp.yaml"),
    overrides={"training": {"epochs": 200}},
)
config.resolve()
config.freeze()

# 定义、注册组件
@model.register("vit_base")
class VisionTransformer(nn.Module):
    config_schema = {
        "image_size": {"type": int, "default": 224},
        "patch_size": {"type": int, "default": 16},
        "dim": {"type": int, "default": 768},
        "depth": {"type": int, "default": 12},
        "heads": {"type": int, "default": 12},
        "dropout": {"type": float, "default": 0.1},
    }
    def __init__(self, ...):
        super().__init__()
        ...

@loss.register("cross_entropy")
class CrossEntropyLoss(nn.Module):
    config_schema = {
        "weight": {"type": float, "default": 1.0},
    }
    def __init__(self, ...):
        super().__init__()
        ...

# 参数校验
model_params = config.get_namespace("model")
loss_params = config.get_namespace("loss")

model_cls = model.get(model_params["name"])
loss_cls = loss.get(loss_params["name"])

validate_schema(model_params, model_cls.config_schema)
validate_schema(loss_params, loss_cls.config_schema)

# 组件实例化
vit = model_cls(**model_params)
cross_entropy = loss_cls(**loss_params)

# 配置快照
checkpoint = config.to_dict()
```

---

## 六、导入

```python
from nameframe.config import (
    Config,
    ConfigConflictError,
    FrozenConfigError,
    validate_schema,
)
```

---
