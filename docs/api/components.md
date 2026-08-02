# 组件系统 API

> `nameframe/components` 定义模型、数据集、损失、指标四种组件。各组件间只通过不可变的 `Batch` 对象通信。内部依赖为 `nameframe.utils` 和 `nameframe.config`，外部依赖为 *Python 3.10+* 标准库、*PyTorch 2.0+*。

---

## 一、全局入口

`docs/api/base.md` 和 `docs/api/config.md` 分别给出了组件注册和组件配置流程。有了组件定义和配置实例后，就要考虑如何给不同组件注入参数，将其实例化了。`nameframe/components` 提供对数据、模型、损失、指标等组件的实例化方法。其导入方法为：

```python
from nameframe.components import (
    Batch,
    Dataset,
    LossProtocol,
    Metric,
    ModelProtocol,
    from_config,
)
```

---

## 二、`Batch` 数据对象

模型、损失函数、指标之间传递数据的格式有时是 `(data, target)` 元组，有时是字典（`dict`），有时是数据类（`dataclass`）。另外，图像、文本、音频等不同模态数据在处理和模型输入方面差异非常大。为增强复用性，需要规范组件间数据传递的格式。

在定义各种组件前，先将组件间的通信格式规整为 `Batch` 数据类。该类不对数据格式做任何假设，因而可广泛用于各种任务。所有组件只接受 `Batch`，只产出 `Batch`（或基于 `Batch` 派生的 `Tensor`）。需注意该数据类负责组件间的通信，处于底层，一般情况下用户不应知道其存在。

### 2.1 数据字段

```python
@dataclass(frozen=True)
class Batch:
    data: Tensor  # 输入
    target: Tensor | None = None  # 标签
    meta: dict[str, Any] = field(default_factory=dict)  # 元数据
```

**`data`** 存储模型的直接输入。例如，图像是高维张量，文本是 token id 序列，多模态是自行约定的嵌套结构。

**`target`** 存储标签，也就是所谓的 *ground truth*。推理模式和无监督任务中，`target` 总是 `None`。

**`meta`** 存储元数据，例如样本文件名、原始文本、数据来源标记等。

### 2.2 不可变

任何改变 `Batch` 的尝试都会抛出 `FrozenInstanceError`。一个 `Batch` 从 `__getitem__` 产出后，其内容应该在训练中不变。

```python
batch.data = torch.randn(2, 10)
batch.target = None
batch.meta = {"new": "dict"}
```

### 2.3 `to(device)` 方法

`Batch` 提供 `to(device)` 方法，该方法与 PyTorch 的 `Tensor.to()` 行为一致。也就是说，将数据的 Tensor 变量 copy 一份到指定设备上去。需注意，应该 **只在最初** 调用 `.to(device)`。对 `Batch` 实例来说，从模型传入到损失再到指标，不应经过任何第二次设备转移。

```python
def to(self, device: torch.device | str) -> Batch
```

例如，传入模型前调用 `to` 方法，后续组件传入的 `batch` 已在正确设备上：

```python
batch = batch.to("cuda:0")
model(batch)
loss(pred, batch)
```

---

## 三、`from_config` 实例化函数

按照约定，任何定义了 `config_schema` 属性的普通类都可以通过 `from_config` 函数进行实例化。

### 3.1 参数

```python
def from_config(cls: type, config: Config, namespace: str) -> Any
```

- **`cls`** 要实例化的 **类** 本身。调用方负责从注册表查找后再传入。
- **`config`** 配置树实例。
- **`namespace`** 配置树中的命名空间，如 `"training.optimizer"`。传给 `config.get_namespace()`。

### 3.2 构造过程

例如，我们有命名空间 `"model"`：

```yaml
model:
  name: resnet50  # 注册表 key
  num_layers: 12  # 构造参数
  hidden_dim: 768 # 构造参数
```

执行 `from_config` 函数，该函数首先提取 `"model"` 下的所有键值对，得到一个字典。然后，移除字典中不应进入构造函数的 `"name"` 键。其次，函数获取目标类的 schema 声明，调用 [配置系统](config.md) 给出的 `validate_schema()` 函数检验参数。最后，将参数字典解包，传入构造函数 `cls(**params)`。

### 3.3 参数穿透

`from_config` 获取 schema 声明、校验参数的实现如下：

```python
def from_config(cls: type, config: Config, namespace: str) -> Any:
    ...

    raw_schema: dict = getattr(cls, "config_schema", {})
    schema = {k: v for k, v in raw_schema.items() if k != "name"}
    if schema:
        validate_schema(params, schema)

    return cls(**params)
```

若要构造的类并没有 `config_schema` 属性，那么所有参数都会跳过参数校验，直接传入构造函数。这意味着该函数 **可构造** 未实现 `config_schema` 的组件（如第三方 `nn.Module`）。另外，`validate_schema` 只校验在 `config_schema` 中声明的参数，不在 `config_schema` 中的参数也 **不会被丢弃**，照常传入构造函数。这些情况下，参数会直接 **穿透** 校验流程。下面是一个穿透了校验环节的例子：

```python
class ThirdPartyWrapper(nn.Module):
    def __init__(self, pretrained=True, **kwargs): ...


# 无 config_schema
model = from_config(ThirdPartyWrapper, config, "model")
```

### 3.4 应用例

下面的例子结合了注册表、配置系统和 `from_config`，展示如何从配置文件中解析参数并实例化组件。

```python
from pathlib import Path
from nameframe.registry import model
from nameframe.config import Config
from nameframe.components import from_config
import torch.nn as nn


# 定义模型
@model.register()
class ResNet(nn.Module):
    config_schema = {
        "name": {"type": str, "help": "registry key."},
        "num_layers": {"type": int, "default": 50, "range": (18, 152)},
        "num_classes": {"type": int, "default": 1000},
        "pretrained": {"type": bool, "default": False},
    }

    def __init__(self, num_layers=50, num_classes=1000, pretrained=False):
        super().__init__()
        self.num_layers = num_layers
        self.backbone = build_resnet_backbone(num_layers, num_classes, pretrained)

    def forward(self, batch):
        return self.backbone(batch.data)


# 构造配置
config = Config.from_yaml(Path("exp.yaml"))
config.resolve()
config.freeze()

# 实例化
cls = model.get(config.model.name)  # 用注册表方法查找类名
model_A = from_config(cls, config, "model")  # 构造
```

`model_A = from_config(cls, config, "model")` 等价于

```python
ResNet(num_layers=50, num_classes=1000, pretrained=False)
```

但用户无需手动查找类名、解包并校验参数，只要在配置文件中修改 `model.name`，就能切换不同模型。用户一般无需实现该流程，框架的编排层通过 `nfm train` 封装了配置解析和组件实例化。

### 3.5 区分 `BatchProtocol`

`BatchProtocol` 是一个因信称义的协议，对接给训练管线充当 **类型标注**，使管线不需关心传入参数除结构以外的一切细节；而 `Batch` 类则对接给模型、损失等各种组件，约束其 I/O 行为。

---

下面是几个核心组件。

## 四、`Dataset` 数据集组件

> 处理数据样本的代码往往会变得杂乱且难以维护；理想情况下，我们希望将数据集代码与模型训练代码解耦，以获得更好的可读性和模块化。[1]

在 PyTorch 框架内，无论实际场景的数据格式如何，其都应该被实例化为 `torch.utils.data.Dataset` 对象。 PyTorch 的 `Dataset` 类将各种数据集的要素抽象出来，访问数据集具体结构的逻辑交由用户实现。在该类的基础上，`nameframe.components.Dataset` 进一步封装了围绕数据集的增强、标识、配置等功能。用户只需继承 `Dataset`，并按 `torch.utils.data.Dataset` 一致的要求实现 `__getitem__` 和 `__len__` 方法，即完成数据端配置。

### 4.2 属性

`Dataset` 继承了 `torch.utils.data.Dataset`，自然有其父类的 `__getitem__`、`__getitems__` 和 `__len__` 方法。此外，延申定义了以下属性。

### 4.2.1 类属性 `config_schema`

先前文档关于配置校验和注入的部分充分说明了 `config_schema` 的作用。这是每种组件都有的类属性，也是每个用户自定义组件必须覆写的部分。对于 `nameframe.components.Dataset` 的 `config_schema`，它声明了数据集必须的两个配置项：

- `path: str` 是数据集根目录；
- `augment: bool = True` 控制数据增强 `_transforms` 是否生效。

`config_schema` 有便捷语法。举个例子，在定义实际的数据集子类时，可用解包语法：

```python
from nameframe.register import dataset
from nameframe.components import Dataset


@dataset.register()
class MyDataset(Dataset):
    config_schema = {**Dataset.config_schema, "new_field": {...}}
    ...
```

### 4.2.2 类属性 `_transforms`

`_transforms: list[Callable]` 列表规定了数据变换的集合。这些变换只在训练模式（`nfm train`）下生效。需注意这个列表的内容物是 `Callable`，也就是函数、方法、类这些可调用对象的集合。

### 4.3.3 实例属性 `path` 和 `augment`

**`path`** 和 **`augment`** 由 `from_config` 或手动的构造函数传入。`path` 被 `fingerprint()` 用于扫描文件列表，`augment` 在 `transforms()` 中被读取以决定是否拼接 `_transforms`。

### 4.3 方法

### 4.3.1 `fingerprint()`

```python
def fingerprint(self) -> str
```

`fingerprint` 基于数据返回哈希值。该方法在训练启动时自动执行，将该哈希值写入该次实验的元数据。对于无法用文件列表描述的场景，比如远程数据、程序生成数据和从 SQL 读取的流式数据，这些场景的 `Dataset` 子类应根据具体情况覆写该方法。

### 4.3.2 `_base_transforms()`

有些变换无论是训练、验证还是推理，都必须以相同参数、相同顺序执行，比如 `ToTensor` 和 `Normalize`。`_base_transforms` 返回这种通用的变换序列。自定义数据集子类时，可以通过覆写该方法来定义这些通用的变换。下面是覆写实例：

```python
def _base_transforms(self) -> list[Callable]:
    return [
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
```
### 4.3.3 `transforms(mode)`

另外一些变换，比如 `RandomCrop`、`RandomHorizontalFlip`、`ColorJitter` 等，应该只对训练集生效。`transforms(mode)` 可以根据参数 `mode` 返回针对某数据集的变换序列。`mode` 只有 `"train"` 和 `"infer"` 两个合法值。

```python
def transforms(self, mode: str = "train") -> list[Callable]
```

另外，`Dataset.__init__` 中读取的 `augment` 实例属性也能覆盖模式行为，方便消融实验。在配置文件中修改 `augment: false` 即可关闭所有增强。

### 4.3.4 魔法方法

PyTorch `Dataset` 规定了子类必须自己实现的两个魔法方法：`__getitem__` 和 `__len__`。其 `__getitem__` 方法的返回值类型是 `Any`，但 `nameframe.components.Dataset` **必须返回 `Batch`**。

```python
def __getitem__(self, index: int) -> Batch
def __len__(self) -> int
```

**1. `__getitem__`**

`__getitem__` 需实现 **用索引获取数据**。这个过程需用户基于数据集结构自定义处理逻辑。让我们看看原生类的字符串文档怎么说。

```python
# torch/utils/data/dataset.py
class Dataset(Generic[_T_co]):
    r"""
    An abstract class representing a :class:`Dataset`.
    
    All datasets that represent a map from keys to data samples should subclass it.
    
    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a data sample for a given key.
    ...
    """
    def __getitem__(self, index) -> _T_co:
    raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")
```

以下是一个简单的例子：

```python
from pathlib import Path


class CatsVsDogs(Dataset):
    def __init__(self, path: str):
        self.path = path
        root = Path(path)
        self._files = sorted(root.glob("*.jpg"))
        self._labels = [0 if "cat" in f.name else 1 for f in self._files]

    def __getitem__(self, index: int) -> Batch:
        img = Image.open(self._files[index]).convert("RGB")
        tensor = transforms.ToTensor()(img)
        return Batch(
            data=tensor,
            target=torch.tensor(self._labels[index]),
            meta={"path": str(self._files[index])},
        )
```

**2. `__len__`**

`__len__` 方法应返回数据集的大小，即样本的数量。同样需根据数据集结构自定逻辑。PyTorch `DataLoader` 依赖它构建索引采样。

**3. `__getitems__`**

除必须实现的 `__getitem__` 和 `__len__` 外，PyTorch 还规定了可选实现的 `__getitems__` 方法。该方法传入多个索引，返回多批数据，加速数据 I/O。`nameframe.components.Dataset` 继承了该方法，返回类型是 `list[Batch]`。

```python
def __getitems__(self, indices: list[int]) -> list[Batch]:
    return [self.__getitem__(i) for i in indices]
```

### 4.4 应用例

延申一下上面的例子，形成一个处理图像分类数据集的完整流程：

```python
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch
from nameframe.registry import dataset
from nameframe.components import Dataset, Batch


@dataset.register()
class CatsVsDogs(Dataset):
    config_schema = {
        **Dataset.config_schema,
        "image_size": {"type": int, "default": 224, "range": (32, 1024)},
    }

    # 训练集增强
    _transforms = [
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ]

    def __init__(self, path: str, augment: bool = True, image_size: int = 224):
        self.path = path
        self.augment = augment
        self.image_size = image_size
        root = Path(path)
        self._files = sorted(root.glob("*.jpg"))
        self._labels = [0 if "cat" in f.name else 1 for f in self._files]

    # 必须覆写
    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, index: int) -> Batch:
        img = Image.open(self._files[index]).convert("RGB")
        # 训练集变换
        pipeline = self.transforms(mode="train")
        tensor = transforms.Compose(pipeline)(img)
        return Batch(
            data=tensor,
            target=torch.tensor(self._labels[index]),
            meta={"path": str(self._files[index])},
        )

    # 通用变换
    def _base_transforms(self) -> list:
        return [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
```

`transforms(mode)` 涵盖训练、验证、推理三种场景的行为差异：

```python
ds = CatsVsDogs(path="/data/cats_vs_dogs")

# 训练
for batch in DataLoader(ds, batch_size=114514):
    # transforms(mode="train")
    #    -> base + RandomResizedCrop + Flip + ColorJitter
    ...

# 验证
ds_val = CatsVsDogs(path="/data/cats_vs_dogs_val", augment=False)
# transforms(mode="train") with augment=False
#    -> base only

# 推理自动调用 transforms(mode="infer")
#    -> base only
```

---

## 五、`ModelProtocol` 模型组件

我们希望尽量简单地定义模型。最好的方法莫过于大家都在用的、继承 `nn.Module` 的方式。不过，我们需要在基类基础上做一些功能延伸和数据接口。最直接的想法是让 `BaseModelInNameFrame` 基类实现延伸功能，让自定义模型双继承，比如：

```python
class MyModel(nn.Module, BaseModelInNameFrame): ...
```

但这样会引入新基类 `BaseModelInNameFrame`，用户体验不简洁。Python Typing 库提供了一种 “因信称义” 的基类 [`Protocol`](https://juejin.cn/post/7113926906407288869)。某类型只要有该 Protocol 的结构，就被认为是该 Protocol 的 “子类”。`ModelProtocol` 约定了模型类组件应有的方法和属性，而用户 **不需** 显式继承 `ModelProtocol`。于是，我们可以沿用继承 `nn.Module` 的方式简单地定义模型：

```python
class MyModel(nn.Module): ...
```

### 5.1 协议要求

想变成 `ModelProtocol` 的一员，自定义模型必须实现以下方法和属性：

```python
class ModelProtocol(Protocol):
    config_schema: dict[str:Any]

    def forward(self, batch: Batch) -> Tensor: ...
    def parameters(self) -> Iterator[Parameter]: ...
    def state_dict(self) -> dict[str:Any]: ...
    def load_state_dict(
        self, state_dict: dict[str:Any], strict: bool = True
    ) -> None: ...
```

### 5.2 属性

`ModelProtocol` 的 `config_schema` 类属性声明模型的构造参数和默认值。框架在实例化模型时会读取该属性，校验配置文件中传入的参数，并将其传入构造函数。

### 5.3 方法

#### 5.3.1 `forward` 类方法

和继承自 `nn.Module` 的 `forward` 方法的唯一区别是，其输入、输出都是 `Batch` 而非 `Tensor`。`forward` 方法应从 `Batch.data` 获取输入。

#### 5.3.2 其他方法

`parameters()`、`state_dict()`、`load_state_dict()` 等方法，在 `nn.Module` 已提供标准实现。除非模型有特殊的状态管理需求（如动态层），否则无需覆写。

### 5.4 应用例

```python
import torch
import torch.nn as nn
from torch.nn import Parameter
from nameframe.registry import model
from nameframe.components import Batch


@model.register()
class VisionTransformer(nn.Module):
    config_schema = {
        "name": {"type": str, "help": "registry key."},
        "image_size": {"type": int, "default": 224, "range": (32, 1024)},
        "patch_size": {"type": int, "default": 16, "choices": [8, 16, 32]},
        "num_classes": {"type": int, "default": 1000},
        "dim": {"type": int, "default": 768, "range": (128, 4096)},
        "depth": {"type": int, "default": 12, "range": (1, 128)},
        "heads": {"type": int, "default": 12, "choices": [6, 8, 12, 16]},
        "dropout": {"type": float, "default": 0.1, "range": (0.0, 0.5)},
    }

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

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = Parameter(torch.randn(1, 1, dim))
        self.pos_embed = Parameter(torch.randn(1, num_patches + 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, batch: Batch) -> torch.Tensor:
        # 从 batch.data 获取张量
        # (B, 3, H, W) -> (B, D, h, w)
        x = self.patch_embed(batch.data)
        x = x.flatten(2).transpose(1, 2)  # (B, D, N)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)[:, 0]  # 取 CLS token 输出
        return self.mlp_head(x)  # (B, num_classes)
```

可见 `ModelProtocol` 对优化用户体验有很大帮助。这个 ViT 定义和大家都在用的 `nn.Module` 只有 `config_schema` 和 `forward` 参数类型两个差异，其余风格与 `nn.Module` 完全一致。

---

## 六、`LossProtocol` 损失函数组件

PyTorch 框架下的损失其实也是 `nn.Module`。因此我们的损失组件采用和模型组件相同的因信称义策略。损失函数直接继承 `nn.Module`，声明 `config_schema`，在 `forward` 中同时接收 `pred` 和 `batch`，就可与其余组件交互。

### 6.1 协议要求

```python
class LossProtocol(Protocol):
    config_schema: dict

    def forward(self, pred: Tensor, batch: Batch) -> Tensor: ...
```

`forward` 的两个参数为什么不是 `forward(pred, target)` 而要多传整个 `batch`？因为我们需要充分利用 `Batch` 类提供的接口。某些情况下，损失函数需要访问 `batch.data`（如知识蒸馏中的教师输出）、`batch.meta`（如样本加权损失）或 `batch.target`（如标签平滑需在损失内部修改标签）。

损失需迎合 PyTorch 的自动微分器和梯度图，所以返回是 **标量** `Tensor`。

### 6.2 应用例

下面的例子展示了如何实现一个带标签平滑的加权交叉熵损失函数。所有要点是继承 `nn.Module`，声明 `config_schema`，`forward` 接收 `pred` 和 标签 `batch`，并返回标量损失。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from nameframe.registry import loss
from nameframe.components import Batch


@loss.register()
class WeightedCrossEntropy(nn.Module):
    config_schema = {
        "name": {"type": str, "help": "registry key."},
        "label_smoothing": {"type": float, "default": 0.0, "range": (0.0, 0.5)},
        "reduction": {
            "type": str,
            "default": "mean",
            "choices": ["mean", "sum", "none"],
        },
    }

    def __init__(self, label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, pred: Tensor, batch: Batch) -> Tensor:
        # 直接从 batch 中获取 target
        target = batch.target
        n_classes = pred.size(-1)

        # 标签平滑
        smooth_target = torch.full_like(pred, self.label_smoothing / (n_classes - 1))
        smooth_target.scatter_(-1, target.unsqueeze(-1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(pred, dim=-1)
        loss_per_sample = -(smooth_target * log_probs).sum(dim=-1)

        # 直接从 batch 中获取元信息
        if "weight" in batch.meta:
            loss_per_sample = loss_per_sample * batch.meta["weight"].to(
                loss_per_sample.device
            )

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        return loss_per_sample  # "none"
```

借助 `typing.Protocol` 实现的 `ModelProtocol` 和 `LossProtocol` 有比较丝滑的使用体验，非常轮椅。

---

## 七、`Metric` 指标组件

指标的计算是**累积**的。每个批次（`Batch`）结束后，应 **累积** 指标状态；每轮训练（epoch）结束后，应 **计算** 指标；新的轮次开始前，需要 **重置** 指标状态。指标组件 `Metric` 的实现类似于 `nameframe.components.Dataset`，用户需继承并覆写 `update(pred, batch)`、`compute()` 和 `reset()` 三个方法。

```python
class Metric:
    def update(self, pred: Tensor, batch: Batch) -> None:
        raise NotImplementedError

    def compute(self) -> dict[str, float]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
```

### 7.1 `update`

该方法应该实现 **计算某 batch 的指标** 的逻辑。`pred` 是模型输出的张量预测，`Batch` 是对应的输入。与 `forward` 方法类似，这里也直接传入了 `Batch`，可以在 `update` 中根据需要访问 `Batch.data`、`Batch.target` 和 `Batch.meta`，计算指标状态。每个批次（`Batch`）结束后，应自动或手动调用此方法，计算并返回当前 `Batch` 的指标结果。

### 7.2 `compute`

该方法应该实现 **汇总一轮内全部 batch 的指标** 的逻辑，并返回 `dict[str, float]` 类型字典。在每轮（epoch）结束时，应自动或手动调用此方法，计算并返回当前轮次的指标结果。

### 7.3 `reset`

该方法应该实现 **初始化内部变量并重置其值** 的逻辑。在每个 epoch 开始前，应自动或手动调用此方法，将指标状态重置为初始值。

### 7.4 应用例

下面实现一个多分类任务指标 micro-F1。

```python
import torch
from torch import Tensor
from nameframe.registry import metric
from nameframe.components import Metric, Batch


@metric.register()
class MultiClassF1(Metric):
    """输出 Precision、Recall、F1。"""

    config_schema = {
        "name": {"type": str, "help": "registry key."},
        "num_classes": {"type": int, "default": 10, "range": (2, 10000)},
    }

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.reset() # 初始重置

    # 必须覆写
    def update(self, pred: Tensor, batch: Batch) -> None:
        """累积某 batch 的混淆矩阵"""
        pred_label = pred.argmax(dim=-1)
        target = batch.target
        for c in range(self.num_classes):
            self._tp[c] += ((pred_label == c) & (target == c)).sum().item()
            self._fp[c] += ((pred_label == c) & (target != c)).sum().item()
            self._fn[c] += ((pred_label != c) & (target == c)).sum().item()

    def compute(self) -> dict[str, float]:
        """汇总全部 batch 的指标"""
        tp = sum(self._tp.values())
        fp = sum(self._fp.values())
        fn = sum(self._fn.values())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        return {"precision": p, "recall": r, "f1": f1}

    def reset(self) -> None:
        """重置计数"""
        self._tp = {c: 0 for c in range(self.num_classes)}
        self._fp = {c: 0 for c in range(self.num_classes)}
        self._fn = {c: 0 for c in range(self.num_classes)}
```

框架的编排层通过 `nfm train` 封装了指标管理，用户只需把指标实例传给训练器 `Trainer` 即可自动维护。若需自己编排相关流程，需要手动按顺序调用方法：

```python
from nameframe.config import Config
from nameframe.registry import metric
from nameframe.components import from_config

# 配置注入
config = Config.from_yaml("exp.yaml").resolve().freeze()
cls = metric.get(config.metric.name)
f1_metric = from_config(cls, config, "metric")

for epoch in range(num_epochs):
    # epoch 开始
    metric.reset()
    ...
    for batch in dataloader:
        pred: torch.Tensor = model(batch)
        loss: torch.Tensor = loss_fn(pred, batch)
        # 每个 batch 累积指标
        metric.update(pred, batch)
    ...
    # Epoch 结束
    result: dict[str, float] = metric.compute()
```

### 7.5 区分 `MetricProtocol`

`MetricProtocol` 是一个因信称义的协议，对接给训练管线充当 **类型标注**，使管线不需关心传入参数除结构以外的一切细节；而 `Metric` 类则提供给用户使用。知道其底层细节后，用户实际上也可以 “继承” `MetricProtocol`，这样就可在不显式继承任何组件的情况下自定义指标，但需要自己维护内部状态。例如：

```python
class MyMetric(MetricProtocol):
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.reset()

    def update(self, pred: Tensor, batch: Batch) -> None:
        ...

    def compute(self) -> dict[str, float]:
        ...

    def reset(self) -> None:
        ...
```

---

## 八、导入

```python
from nameframe.components import (
    Batch,
    Dataset,
    LossProtocol,
    Metric,
    MetricProtocol,
    ModelProtocol,
    from_config,
)
```

`ModelProtocol` 和 `LossProtocol` 也可以从 `nameframe.utils` 导入。

---

References：

- [1] [pytorch 文档](https://docs.pytorch.ac.cn/tutorials/beginner/basics/data_tutorial.html)