# 基础 API

> `nameframe.registry` 提供组件注册与查找，`nameframe.utils` 提供类型协议、随机数种子和环境记录工具。

## 一、组件注册

模型、数据集等组件可以直接用 Python 构造。需要根据配置中的名称切换组件时，再使用注册表：先保存名称与类的对应关系，使用时按名称查找，最后传入参数构造实例。

### 1.1 注册与构造

```python
from torch import nn

from nameframe.registry import Registry

models = Registry("models", strict=True)
models.register_external("linear", nn.Linear)

cls = models.get("linear")
classifier = cls(in_features=4, out_features=2)
assert classifier.in_features == 4
```

`register_external(name, item)` 用于注册已有的类或可调用工厂。自己定义组件时，也可在定义处加上 `@models.register("linear")`。省略名称则自动将 CamelCase 转为 snake_case；没有函数名的可调用实例需要显式命名。注册名必须非空，且不能含 `:`。

注册只保存对应关系，构造仍由用户完成。上例得到的 `nn.Linear` 接收 Tensor；接入 [Batch](components.md) 数据链路时，需要从 `batch.data` 取出输入。

项目可以自建注册表，也可使用 `nameframe.registry` 提供的四个全局实例：

| 实例 | 用途 |
|------|------|
| `model` | 模型 |
| `dataset` | 数据集 |
| `loss` | 损失函数 |
| `metric` | 评估指标 |

`Registry(name, base_type=None, *, strict=False)` 的 `base_type` 用于限制注册类型。指定后，只接受该基类的子类，不符合时抛出 TypeError。

### 1.2 重名与命名空间

不同定义使用同一个名称，可能意外覆盖已有组件。新项目建议使用 `strict=True`，此时重名会抛出 RegistryConflictError，并列出新旧定义。默认 `strict=False` 允许覆盖；注册时传入 `replace=True` 或 `replace=False`，可单独允许或禁止覆盖。同一个对象重复注册不报错。

需要保留多个同名组件时，可用 `create_namespace(name)` 创建子表，再通过 `namespace:key` 查找。子表继承父表的类型限制和 strict 设置。

| 方法 | 说明 |
|------|------|
| `get(name)` | 查找名称或 `namespace:key`，返回类或工厂 |
| `create_namespace(name)` | 创建或获取子表 |
| `list_all(namespace=None)` | 返回当前表或指定子表的映射副本 |
| `conflicting_keys(namespace)` | 列出当前表与指定子表的同名键 |

注册表还支持 `in` 和 `len()`。查找失败抛出 RegistryKeyError，可通过其 `key`、`registry_name`、`available` 属性查看请求名称、表名和可用名称。

## 二、按目录发现组件

注册代码在模块被导入时执行，因此通常只需显式导入组件所在模块。需要批量导入目录时，可调用 `discover(directory, strict=True)`；它会扫描 Python 模块，包括包的 `__init__.py`，并触发其中的注册代码。

成功时返回 DiscoveryReport，包含 `imported` 和 `failures`。每条失败记录保留模块名、文件路径和原始异常。严格模式完成扫描后，若有失败则抛出 DiscoveryError，扫描结果保存在其 `report` 属性中；默认非严格模式只发出警告。

发现过程保留相对导入，成功模块使用导入缓存，失败模块可重试；同名模块会检查文件来源，结束后恢复 sys.path。不过，导入已经执行的注册等操作不会回滚，因此应在构造组件前完成发现。

## 三、组件类型协议

组件之间需要约定输入和输出，但不必继承同一个基类。`nameframe.utils` 提供以下协议，满足对应方法即可用于类型检查：

| 协议 | 方法 |
|------|------|
| `ModelProtocol` | `__call__(batch) -> Tensor` |
| `LossProtocol` | `__call__(pred, batch) -> Tensor` |
| `MetricProtocol` | `update(pred, batch)`、`compute() -> dict[str, float]`、`reset()` |
| `DatasetProtocol` | `__getitem__(index) -> Batch`、`__len__()`、`transforms(mode)` |

协议不要求声明 config_schema。它们描述调用方式，不会在运行时检查张量形状或损失是否为标量。数据字段的实际检查见[数据管线](data.md)；配置字段的描述类型 FieldSchema 见[配置系统](config.md)。

## 四、随机种子与环境

```python
from nameframe.utils import capture_env, derive_seed, set_seed

set_seed(42)
model_seed = derive_seed(42, "model_init")
worker_seed = derive_seed(42, "loader", rank=1)
env = capture_env()

assert 0 <= model_seed < 2**32
assert "torch" in env
```

`set_seed(seed)` 接受 `[0, 2**32)` 内的整数，设置 Python、NumPy、PyTorch 的随机源，并启用 cuDNN 确定性模式、关闭 benchmark。`derive_seed(base_seed, component, rank=0)` 根据这三个值派生种子，相同输入得到相同结果。

独立 generator、worker 和其他库的随机源仍需自行设置。固定种子也不保证不同版本、设备和算法之间逐位一致。

`capture_env()` 返回 `python`、`torch`、`cuda`、`cudnn`、`platform` 五个字符串字段；CUDA 不可用时，相关版本为 `"none"`，可将结果保存到实验记录中。
