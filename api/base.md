# 底层基础 API

> `nameframe/utils` 和 `nameframe/registry` 为上层框架提供方便的组件注册、模块间共享数据类型的通信协议，以及随机数种子、环境管理等小工具。
> 底层是零内部依赖的，外部仅依赖 *Python 3.10+* 标准库和 *PyTorch 2.0+*。

---

## 一、全局入口

包的顶层提供三个属性：

| 符号 | 类型 | 说明 |
|------|------|------|
| `__version__` | `str` | 版本号 |
| `registry` | module | 提供全局注册实例 `Registry` |
| `utils` | module | 通信协议和种子、环境等零碎小工具 |

```python
from nameframe import __version__, registry, utils
```

---

## 二、`nameframe.registry` 注册器

为方便用户拓展新的组件，采用 **注册表** 管理包括模型、损失等在内的所有组件。框架本身只实现基于注册表的组件管理和调用，将组件的注册交给用户，采用 **各自注册** 的方式拓展组件。例如，用户新定义一种损失函数，只需在定义处对该损失函数简单注册，然后在配置文件中修改损失的配置项，即可切换自定义的损失函数，不需要做任何代码上的修改。那么，如何注册组件呢？

### 2.1 全局 Registry 单例

首先需要导入注册表。注册表是以实例的形式存在的。注册表的底层提供这样的接口：每种组件对应一个该类组件的注册表的 **全局单例**，后续所有相关组件可直接从 `nameframe.registry` 导入，即实现注册或查找：

```python
from nameframe.registry import model, dataset, loss, metric, ops, plugin
```

每个实例对应一种组件：

| 实例 | 用途 |
|:------:|:------:|
| `model` | 模型架构 |
| `dataset` | 数据集 |
| `loss` | 损失函数 |
| `metric` | 评估指标 |
| `ops` | 自定义算子 |
| `plugin` | 拓展插件 |

### 2.2 `Registry` 类

注册表是组件系统的核心，管理变组件名（`str`）到类（`type`）的映射。每个 `Registry` 实例允许创建命名空间，实现项目间/类型间的隔离。

#### 构造

```python
Registry(name: str, base_type: type | None = None)
```

| 参数 | 类型 | 说明 |
|:------:|:------:|:------:|
| `name` | `str` | 组件名 |
| `base_type` | `type \| None` | 可选，表示新的基类 |

#### 注册组件

三种方式都可注册组件：

**1. 装饰器注册**

```python
# 导入 model 全局单例
from nameframe.registry import model

# 调用 model 装饰器方法注册
@model.register("my_vit")
class MyViT(nn.Module):
    ...

# 缺省时自动补全: MyViT -> "my_vit"
@model.register()  
class MyViT(nn.Module):
    ...
```

`register(name: str | None = None) -> Callable` 是一个装饰器工厂。`name` 为 `None` 时，自动将 CamelCase 格式的类名转换为 snake_case 作为 `name`（键名）。返回的装饰器不侵入原类，注册后原样返回。这么好的方法，建议多用。

**2. 函数注册**

通过每个类别单例的内置函数 `register_external` 注册，其语法如下。该方法用于方便注册第三方库中的类。

```python
register_external(name: str, item: type) -> None
```

应用例：

```python
model.register_external("resnet50", torchvision.models.resnet50)
```

**3. 自动发现**

若不确定某路径下有多少要注册的组件，框架提供了自动搜索注册的方法：

```python
discover(Path("my_project/components"))
```

`discover(directory: Path) -> None` 递归扫描目录下所有 `.py` 文件（跳过 `__init__.py`），导入每个模块以对其尝试触发 `@*.register`。注册失败时触发 `Warning`。

> NOTE：如果构造时传入 `base_type`，三种方式都会在注册时做 `issubclass` 校验，不满足则抛出 `TypeError`。

#### 查找组件

不同组件的注册表单例还提供一些管理和查找的接口。

```python
# 直接查找
cls = model.get("my_vit")

# 命名空间查找
cls = model.get("proj_a:my_vit")

# 成员检测
"my_vit" in model   # -> True / False

# 列出所有
model.list_all()
# -> {"my_vit": <class MyViT>, ...}

model.list_all(namespace="proj_a")
# -> proj_a 下的所有映射
```

`get(name: str) -> type` 根据输入值检索注册表，支持 `"ns:key"` 格式的命名空间。未找到会抛出 `RegistryKeyError(key, registry_name, available_list)`。

`list_all(namespace: str | None = None) -> dict[str, type]` 返回所有已注册映射。指定 `namespace` 时只返回该命名空间内的注册。

#### 魔法方法

此外，还有一些魔法方法，例如注册表支持 `in` 关键字和 `len()` 方法：

```python
# bool
"my_vit" in model

# len of model registry
model.len()
```

#### 命名空间隔离

遇到不同项目中的同名注册（如 `resnet`）时，可用命名空间隔离：

```python
proj_a = model.create_namespace("proj_a")
proj_a.register("resnet")(ResNetA)

proj_b = model.create_namespace("proj_b")
proj_b.register("resnet")(ResNetB)

model.get("proj_a:resnet") # ResNetA
model.get("proj_b:resnet") # ResNetB
```

`create_namespace(name: str) -> Registry` 创建或获取子注册表。子注册表继承父注册表的 `base_type`。

`conflicting_keys(namespace: str) -> list[str]` 调试用。检查当前注册表中与指定命名空间重复的 key。

#### 异常

`RegistryKeyError(KeyError)` 在 `get()` 方法查找失败时抛出，其三个属性说明查找的注册名在某注册表中不存在，并列出存在的注册：

| 属性 | 类型 | 说明 |
|------|------|------|
| `key` | `str` | 请求的 key |
| `registry_name` | `str` | 注册表名称 |
| `available` | `list[str]` | 当前可用的 key 列表 |

组件注册的全部内容如上。

---

## 三、`nameframe.utils` 的数据和配置交互

### 3.1 `BatchProtocol` 数据交互

预处理管线、模型内部、损失、推理等各组件之间的数据通信没有约束，管理起来很乱。框架定义 `BatchProtocol` 作为组件数据交互的基本类型。该类对所有组件之间传递数据的结构做约定：

```python
class BatchProtocol(Protocol):
    data: Tensor           # 模型输入
    target: Tensor | None  # 目标张量，无监督时为 None
    meta: dict             # 辅助元信息
```

### 3.2 各组件数据交互

为兼容不同模态的数据，框架应不需知晓数据格式，即可使各模块发挥作用。对此，不同组件应有各自的数据协议类型，定义该组件的数据结构：

| Protocol | 方法 |
|:--------:|:---:|
| `ModelProtocol` | `forward(batch) -> Tensor`、`parameters()`、`state_dict()`、`load_state_dict()` |
| `LossProtocol` | `forward(pred, batch) -> Tensor` |
| `MetricProtocol` | `update(pred, batch)`、`compute() -> dict[str, float]`、`reset()` |
| `DatasetProtocol` | `__getitem__`、`__len__`、`fingerprint()`、`transforms(mode)` |

> *“我们是因信称义的”*。具体组件 **不需显式继承** 对应协议，只需满足结构即可。

下面是一个 `ModelProtocol` 的例子。应用时，只需从 `nameframe.utils` 导入：

```python
from nameframe.utils import (
    BatchProtocol,
    ModelProtocol,
    LossProtocol,
    MetricProtocol,
    DatasetProtocol,
)
```

第二步，根据 `ModelProtocol` 协议定义模型。查找 `ModelProtocol` 约定的结构，发现模型必须实现一个 `config_schema` 字典和四个 `forward(batch) -> Tensor`、`parameters()`、`state_dict()`、`load_state_dict()` 方法。因而给出以下定义，可使 `VisionTransformer` 类与其余组件、配置自动交互：

```python
...
from nameframe.registry import model
from nameframe.utils import BatchProtocol


@model.register()
class VisionTransformer(nn.Module):
    """满足 ModelProtocol 协议的 ViT 模型"""

    # config_schema
    config_schema = {
        "image_size": {
            "type": int,
            "default": 224,
            "help": "输入图像尺寸",
        },
        "patch_size": {
            "type": int,
            "default": 16,
            "help": "Patch 大小",
        },
        "num_classes": {
            "type": int,
            "default": 1000,
            "help": "分类类别数",
        },
        "dim": {
            "type": int,
            "default": 768,
            "range": (128, 4096),
            "help": "隐藏层维度",
        },
        "depth": {
            "type": int,
            "default": 12,
            "range": (1, 128),
            "help": "Transformer 层数",
        },
        "heads": {
            "type": int,
            "default": 12,
            "help": "注意力头数",
        },
        "dropout": {
            "type": float,
            "default": 0.1,
            "range": (0.0, 0.5),
            "help": "Dropout 概率",
        },
    }

    # 构造函数，参数与 config_schema 逐对应
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        num_patches = (image_size // patch_size) ** 2

        self.cls_token = Parameter(torch.randn(1, 1, dim))
        self.pos_embed = Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dropout=dropout,
                batch_first=True,
            ),
            num_layers=depth,
        )
        self.mlp_head = nn.Linear(dim, num_classes)

    # 四个方法
    def forward(self, batch: BatchProtocol) -> Tensor:
        """要求 forward 方法，输入 BatchProtocol，输出 Tensor."""
        x = self.patch_embed(batch.data) # (B, 3, H, W) -> (B, D, h, w)
        x = x.flatten(2).transpose(1, 2) # (B, D, N)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)[:, 0] # CLS token
        return self.mlp_head(x) # (B, num_classes)

    # 以下同上
    def parameters(self) -> Iterator[Parameter]:
        return super().parameters()

    def state_dict(self) -> dict[str, Any]:
        return super().state_dict()

    def load_state_dict(self, state_dict: dict[str, Any],
            strict: bool = True) -> None:
        super().load_state_dict(state_dict, strict=strict)
```

注意到 `VisionTransformer` 类并没有显式声明自己继承自 `ModelProtocol`，但符合其结构格式，类型检查会将其视为满足 `ModelProtocol`，便可与其余组件和配置项交互。

### 3.3 `config_schema` 和 `FieldSchema` 配置交互

无论何种组件，都有诸多可配置项。可配置项的一端是最顶层的 `.yaml` 文件，面向用户；其另一面深入各组件内部，规定各组件可配置项的元数据。上面见到的 `config_schema` 便是深入组件内部的一端。

各组件的数据类型 xxxProtocol 规定，在每个组件定义时，其内部都必须声明 `dict` 类型的 `config_schema` 变量，规定其所有可配置项元数据。例如：

```python
class MyModel(nn.Module):
    config_schema = {
        "hidden_dim": {
            "type": int,
            "default": 512,
            "range": (64, 2048),
            "help": "隐藏层维度",
        },
        "dropout": {
            "type": float,
            "default": 0.1,
            "range": (0.0, 0.5),
            "nullable": True,
        },
        "activation": {
            "type": str,
            "default": "gelu",
            "choices": ["relu", "gelu", "silu"],
            "help": "激活函数",
        },
    }
```

上面的 `config_schema` 字典，其每个值（`value`）都是一个继承自 `TypedDict` 类型的 `FieldSchema`。该类负责规定一项配置的具体细节，如 `"activation"` 的  `FieldSchema` 规定了可选的激活函数等。`FieldSchema` 的 `total=False` 字段允许只记录其中某些字段。

```python
class FieldSchema(TypedDict, total=False):
    type: type                       # 期望类型
    default: Any                     # 默认值
    help: str                        # 描述文本
    choices: list | None             # 可选值列表
    range: tuple[float, float] | None  # 数值范围 (min, max)
    nullable: bool                   # 是否允许 None，默认 False
```

---

## 四、`nameframe.utils` 的复现工具

### 4.1 种子管理

```python
from nameframe.utils import set_seed, derive_seed

set_seed(42) # 全局种子

# 不同组件派生种子
model_seed = derive_seed(42, "model_init")
loss_seed = derive_seed(42, "dropout")
ddp_seed = derive_seed(42, "dropout", rank=1)
```

`set_seed(seed: int) -> None` 设置所有后端的随机种子（`random`、`numpy`、`torch.cpu`、`torch.cuda`），并强制 cuDNN 确定性模式。

`derive_seed(base_seed: int, component: str, rank: int = 0) -> int` 从主种子为特定组件派生独立种子。相同输入、相同 rank 确保相同的输出。

### 4.2 环境捕获

```python
from nameframe.utils import capture_env

env = capture_env()
# {"python": "3.10.12", "torch": "2.5.1", "cuda": "12.4",
#  "cudnn": "90100", "platform": "Linux-5.15.0-139-generic-..."}
```

`capture_env() -> dict` 返回包含 5 个固定键（`python`、`torch`、`cuda`、`cudnn`、`platform`）的环境签名。无 CUDA 时，`cuda` 和 `cudnn` 返回 `"none"`。

---

## 五、导入

```python
# 入口
from nameframe import __version__

# 注册表
from nameframe.registry import (
    Registry, RegistryKeyError, discover,
    model, dataset, loss, metric, ops, plugin,
)

# 类型协议
from nameframe.utils import (
    BatchProtocol, ModelProtocol, LossProtocol,
    MetricProtocol, DatasetProtocol, FieldSchema,
)

# 工具函数
from nameframe.utils import set_seed, derive_seed, capture_env
```

---
