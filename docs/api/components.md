# 组件系统 API

> `nameframe.components` 提供 Batch 数据对象，以及可选的 Dataset、Metric 基类。模型继续使用普通 PyTorch 模块，各组件通过约定的输入和输出配合。

## 一、Batch 数据对象

数据集、模型、损失和指标都需要访问输入、标签及附加信息。为统一传递方式，NameFrame 将这些内容放在 Batch 中；它既可以表示一个样本，也可以表示拼接后的一个批次。

| 字段 | 类型 | 说明 |
|------|------|------|
| `data` | Tensor | 模型输入，必填 |
| `target` | Tensor 或 None | 标签，默认 None |
| `meta` | dict[str, Any] | 样本编号、来源等信息，默认各自创建空字典 |

```python
import torch

from nameframe.components import Batch

batch = Batch(data=torch.ones(2), meta={"id": 1})
moved = batch.to("cpu")

assert moved is not batch
assert moved.meta == {"id": 1}
assert moved.meta is not batch.meta
```

### 1.1 字段与内容

Batch 使用冻结的数据类，创建后不能重新给字段赋值，例如 `batch.data = other` 会抛 FrozenInstanceError。不过，Tensor 内容和 meta 字典仍能修改。变换需要改变数据时，应返回新 Batch，避免原地修改共享输入。

构造函数本身不检查字段类型。[数据管线](data.md) 在读取样本和执行变换后，检查 data、target 和 meta 是否满足上表约定。其他数据对象接入管线前，应先转换为 Batch。

### 1.2 设备迁移

`to(device, non_blocking=False)` 将 data 和非空 target 移到指定设备；`pin_memory()` 将它们放入锁页内存。两者都返回新 Batch，并浅复制 meta。meta 中的内容不迁移，嵌套对象仍可能共享；同设备迁移也可能复用 Tensor 存储。

通常在主进程取到批次后、调用模型前执行设备迁移，随后将同一个 Batch 传给损失和指标。

## 二、Dataset 数据集

Dataset 继承 PyTorch 的 map-style Dataset。使用时，至少实现 `__getitem__(index) -> Batch` 和 `__len__()`：前者读取原始样本，后者返回样本数。变换交给数据管线执行，避免在读取时重复处理。

### 2.1 读取与通用预处理

下面定义一个小数据集，并将缩放操作放入通用预处理。示例可以独立运行：

```python
from dataclasses import replace

import torch

from nameframe.components import Batch, Dataset


def scale(batch):
    return replace(batch, data=batch.data / 2)


class Samples(Dataset):
    version = "samples-v1"

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return Batch(
            data=torch.tensor([float(index)]),
            target=torch.tensor(index % 2),
            meta={"id": index},
        )

    def _base_transforms(self):
        return [scale]


samples = Samples()
transform = samples.transforms("infer")[0]
assert transform(samples[2]).data.item() == 1.0
assert samples.fingerprint() == "samples-v1"
```

`__getitems__(indices)` 是可选的批量读取方法，默认逐个调用 `__getitem__`，返回 list[Batch]。需要优化批量读取时可覆写，但返回数量和顺序应与索引一致。

### 2.2 按模式组织变换

训练增强只在训练时启用，归一化等通用预处理则通常在训练、推理时都需要。Dataset 用两个工厂分别创建它们：

| 方法 | 用途 |
|------|------|
| `_augmentation_transforms()` | 创建训练增强 |
| `_base_transforms()` | 创建通用预处理 |
| `transforms(mode="train")` | 组合并返回当前模式的变换列表 |

默认训练链先执行增强，再执行通用预处理。mode 为 infer，或实例的 augment 为 False 时，只执行通用预处理。需要其他组合顺序时，可覆写 transforms。

每次创建 loader 都会调用工厂，因此有状态变换应在工厂内新建。将同一对象反复返回，仍会共享状态。无状态函数可以简写在 `_transforms` 中，其列表按数据集实例复制；可调用实例、partial，以及持有可变状态的闭包，应通过工厂创建。

### 2.3 数据版本与配置

默认 `fingerprint()` 返回显式声明的非空 version，未声明时抛 ValueError。数据、拆分或预处理变化时，应同步更新版本。需要内容哈希或外部清单时，可覆写该方法；内核不会自行扫描文件，也不会在加载时强制调用它。

Dataset 基类的 config_schema 声明 path 和 augment，具体构造函数仍由子类实现。通过 from_config 构造时，schema 应与构造参数对应；像上例直接用 Python 构造，则不需要配置系统。

Dataset 也是可选基类。已有数据集只要提供读取、长度和 transforms，就能接入 DataPipeline，无需提供 config_schema 或 fingerprint。

## 三、模型与损失

模型使用普通 nn.Module，在 forward 中读取 Batch，返回预测 Tensor。损失接收预测和同一个 Batch，返回供反向传播的标量 Tensor：

```python
import torch
from torch import nn
from torch.nn import functional

from nameframe.components import Batch


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, batch):
        return self.linear(batch.data)


def cross_entropy(pred, batch):
    assert batch.target is not None
    return functional.cross_entropy(pred, batch.target)


batch = Batch(data=torch.ones(3, 4), target=torch.tensor([0, 1, 0]))
model = Classifier()
loss = cross_entropy(model(batch), batch)
assert loss.ndim == 0
```

应通过 `model(batch)` 调用模型，以保留 PyTorch hooks。张量形状、dtype、标签含义及损失是否为标量，由任务代码保证；ModelProtocol 和 LossProtocol 只提供类型约定，不执行这些检查。

## 四、指标

指标需要保存跨批次的统计状态，可继承 Metric，也可用普通类实现以下三个方法：

| 方法 | 调用时机与用途 |
|------|----------------|
| `reset()` | 一轮开始前，清空统计状态 |
| `update(pred, batch)` | 每个批次后，累计统计量 |
| `compute() -> dict[str, float]` | 一轮结束后，汇总结果 |

训练循环负责按顺序调用这些方法，Metric 基类不自动更新或重置。ModelProtocol、LossProtocol、MetricProtocol 均可从 components 或 utils 导入；组件需要从配置构造时，再接入[配置系统](config.md)。
