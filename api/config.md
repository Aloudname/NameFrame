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

从用户的角度看，无论是超参数、模型结构参数，还是各种各样的杂项参数，其唯一操作入口都是 `.yaml` 配置文件。为了分门别类地将不同去向的参数注入其应有的位置，框架底层将配置项抽象成了一个树。`Config` 是整条管线的唯一入口和唯一出口。

### 2.1 三步生命周期

```
from_yaml()  →  resolve()  →  freeze()
   加载          解析          冻结
   YAML         ${search:...}  只读属性树
```

三者必须**严格按顺序**调用。跳过 `resolve()` 直接 `freeze()`，`${search:...}` 不会被转换为 `SearchSpace`；跳过 `freeze()` 直接注入组件，配置可能在运行时被意外改写。

```python
from pathlib import Path
from nameframe.config import Config

# 第一步：加载 YAML
config = Config.from_yaml(Path("exp.yaml"))

# 第二步：解析变量
config.resolve()

# 第三步：冻结
config.freeze()

# 之后 config 是只读的，可通过属性访问
print(config.model.name)       # → "resnet50"
print(config.training.lr)      # → 0.001
```

### 2.2 构造

```python
Config(data: dict, source_path: Path | None = None)
```

| 参数 | 类型 | 说明 |
|:------:|:------:|:------:|
| `data` | `dict` | 裸配置字典，可手工传入（测试用） |
| `source_path` | `Path \| None` | 文件来源，仅供诊断和相对路径解析 |

多数情况下不直接构造，而是走 `from_yaml()`。

### 2.3 `from_yaml` 加载

```python
@classmethod
def from_yaml(cls, path: Path, overrides: dict | None = None) -> Config:
```

| 参数 | 类型 | 说明 |
|:------:|:------:|:------:|
| `path` | `Path` | `.yaml` 文件路径 |
| `overrides` | `dict \| None` | CLI 传入的覆盖值，深度合并到加载后的配置 |

支持 `!include` 指令，被引用的文件会被整体解析并替换当前节点。`!include` 使用相对路径（相对于当前 YAML 文件所在目录），并有循环引用检测。

示例 `exp.yaml`：

```yaml
model: !include model/resnet50.yaml
training:
  lr: 1e-3
  epochs: 100
  optimizer:
    type: adam
    weight_decay: 1e-4
```

```python
# 基础加载
config = Config.from_yaml(Path("exp.yaml"))

# 带 CLI 覆盖
config = Config.from_yaml(
    Path("exp.yaml"),
    overrides={"training": {"lr": 5e-4}},
)
# 最终 training.lr 是 5e-4，其余 training 字段不变
```

> `!include` 不可与 `<<:` YAML 合并键组合使用（PyYAML 的节点类型判断在 tag 处理器之前）。需要合并语义时，把公共部分抽到一个文件后整体 include。

### 2.4 属性访问

加载后的 `Config` 对象像普通 Python 对象一样通过 `.` 访问，嵌套字典自动转为嵌套属性：

```python
config.model.name          # → "resnet50"
config.training.optimizer.type  # → "adam"
```

支持 `in` 关键字和 `len()`：

```python
"lr" in config.training    # → True
len(config.training)       # → 4（该命名空间下字段数）
```

### 2.5 `resolve` 解析变量

```python
def resolve(self) -> None:
```

将配置树中的所有 `${search: ...}` 占位符转换为 `SearchSpace` 对象。操作为**原地修改**，调用后 `config.training.lr` 原本若是 `"${search: loguniform(1e-5, 1e-2)}"`，将变为 `SearchSpace(expression="loguniform(1e-5, 1e-2)")`。

```python
# exp.yaml 中：
# training:
#   lr: "${search: loguniform(1e-5, 1e-2)}"

config = Config.from_yaml(Path("exp.yaml"))
config.resolve()
type(config.training.lr)   # → <class 'SearchSpace'>
```

### 2.6 `freeze` 冻结

```python
def freeze(self) -> None:
```

递归冻结整棵配置树。冻结后任何写入尝试都会抛出 `FrozenConfigError`，并指明违规代码的文件和行号。

```python
config.freeze()
config.model.name = "vit"  # 抛出 FrozenConfigError
```

### 2.7 `is_frozen` 状态

```python
@property
def is_frozen(self) -> bool:
```

```python
config = Config({"lr": 0.01})
config.is_frozen   # → False
config.freeze()
config.is_frozen   # → True
```

### 2.8 `get_namespace` 提取子命名空间

```python
def get_namespace(self, ns: str) -> dict:
```

| 参数 | 类型 | 说明 |
|:------:|:------:|:------:|
| `ns` | `str` | 点号分隔的路径，如 `"training.optimizer"` |

返回纯 `dict`，用于注入组件构造函数（`**kwargs`）。组件实例化不需要另一层对象包装，`dict` 更轻量也更通用。

```python
# exp.yaml:
# model:
#   backbone:
#     name: resnet50
#     pretrained: true

params = config.get_namespace("model.backbone")
# → {"name": "resnet50", "pretrained": True}
```

### 2.9 `to_dict` 导出

```python
def to_dict(self) -> dict:
```

返回整棵配置树的深拷贝纯 `dict`，供 checkpoint 快照或序列化使用。修改返回的字典不影响原 `Config`。

---

## 三、`validate_schema` 配置校验

配置的值必须通过 `FieldSchema` 规则校验，才能在组件实例化前拦截错误。

```python
def validate_schema(params: dict, schema: dict[str, FieldSchema]) -> None:
```

| 参数 | 类型 | 说明 |
|:------:|:------:|:------:|
| `params` | `dict` | 实际参数值 |
| `schema` | `dict[str, FieldSchema]` | 参数声明 |

### 3.1 五项校验规则

`validate_schema` 按优先级依次检查，**一次报告所有错误**而非遇到第一个就抛：

| 顺序 | 规则 | 违反时行为 |
|:---:|------|------|
| 1 | 未知字段 | 用编辑距离建议 "你是不是想写 X？"；距离太远则列出全部可用字段 |
| 2 | 缺少必填字段 | 字段无 `default` 且 `params` 未提供 |
| 3 | None 合法性 | 字段 `nullable` 不为 `True` 时不可传入 `None` |
| 4 | 类型匹配 | `isinstance(value, declared_type)` 检查 |
| 5 | 值范围 | `choices` 枚举列表匹配或 `range` 数值区间匹配 |

### 3.2 使用示例

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
from nameframe.config import validate_schema, ConfigConflictError
from nameframe.utils.typing import FieldSchema

schema: dict[str, FieldSchema] = {
    "hidden_dim": {"type": int, "default": 512, "range": (64, 2048)},
    "activation": {"type": str, "choices": ["relu", "gelu", "silu"]},
}

# 正常通过
validate_schema({"hidden_dim": 256, "activation": "gelu"}, schema)

# 字段拼错：建议 "activation"
try:
    validate_schema({"hidden_dim": 256, "activatoin": "gelu"}, schema)
except ConfigConflictError as e:
    print(e.errors)
    # → ["unknown field 'activatoin', did you mean 'activation'?"]

# 值超出范围 + 类型不对
try:
    validate_schema({"hidden_dim": "big", "activation": "tanh"}, schema)
except ConfigConflictError as e:
    print(e.errors)
    # → [
    #     "field 'hidden_dim' expects int, got str (value='big').",
    #     "field 'activation' has value 'tanh', but must be one of ['relu', 'gelu', 'silu'].",
    # ]
```

### 3.3 `ConfigConflictError` 异常

```python
class ConfigConflictError(ValueError):
    errors: list[str]  # 所有错误消息，可逐条展示给用户
```

---

## 四、`FrozenConfigError` 冻结保护

`Config.freeze()` 调用后，任何对配置树中任意节点的属性赋值都会抛出该异常，并精确定位违规代码的位置。

```python
class FrozenConfigError(Exception):
    key: str       # 被尝试修改的字段名
    location: str  # 违规代码位置，格式 "path/to/file.py:line"
```

### 使用场景

```python
config = Config.from_yaml(Path("exp.yaml"))
config.resolve()
config.freeze()

# 训练循环中某个回调意外尝试修改配置
def bad_callback(cfg):
    cfg.training.lr = 999  # → FrozenConfigError:
    #   "attempted to change frozen 'lr' at my_callbacks.py:42"
```

---

## 五、`SearchSpace` 搜索空间

`resolver.py` 内部使用的一个轻量 dataclass，供 Phase 8（超参搜索 CLI）消费：

```python
@dataclass
class SearchSpace:
    expression: str  # "${search: ...}" 内部的原始表达式
```

虽然目前未从 `__init__.py` 导出为公共 API，但它是 `resolve()` 的输出产物，后续 Phase 的 sweep 引擎会直接消费它。

---

## 六、完整管线示例

一个端到端流程，串起 Phase 1 的注册表和 Phase 2 的配置系统：

```yaml
# exp.yaml
model: !include model/vit_base.yaml
training:
  lr: "${search: loguniform(1e-5, 1e-2)}"
  epochs: 100
  optimizer:
    type: adam
    weight_decay: 1e-4
```

```yaml
# model/vit_base.yaml
name: vit_base
image_size: 224
patch_size: 16
dim: 768
depth: 12
heads: 12
dropout: 0.1
```

```python
from pathlib import Path
from nameframe.config import Config, validate_schema, ConfigConflictError
from nameframe.registry import model

# 1. 加载 YAML
config = Config.from_yaml(
    Path("exp.yaml"),
    overrides={"training": {"epochs": 200}},
)

# 2. 解析搜索空间
config.resolve()
# config.training.lr 现在是 SearchSpace(expression="loguniform(1e-5, 1e-2)")

# 3. 冻结
config.freeze()

# 4. 提取配置并校验
model_params = config.get_namespace("model")
model_cls = model.get(model_params["name"])
try:
    validate_schema(model_params, model_cls.config_schema)
except ConfigConflictError as e:
    print(f"配置冲突:\n" + "\n".join(f"  - {err}" for err in e.errors))
    raise

# 5. 实例化
vit = model_cls(**model_params)

# 6. 快照
checkpoint_snapshot = config.to_dict()
```

---

## 七、导入速查

```python
# 配置系统
from nameframe.config import (
    Config,
    ConfigConflictError,
    FrozenConfigError,
    validate_schema,
)

# 配合 Phase 1 使用
from nameframe.registry import model
from nameframe.utils.typing import FieldSchema, _field
```

---

## 八、契约与兼容性

以下接口被后续 Phase 依赖，修改需保持向后兼容：

| 接口 | 依赖 Phase | 契约要点 |
|:------|:---:|:------|
| `Config.from_yaml(path, overrides)` | 8 (CLI) | `overrides` 深度合并，支持 `!include` |
| `Config.resolve()` | 8 (CLI) | 将 `${search:...}` 转为 `SearchSpace` |
| `Config.freeze()` | 8 (CLI) | 递归冻结，冻结后写抛 `FrozenConfigError` |
| `Config.is_frozen` | 3, 6 | property，返回 `bool` |
| `Config.get_namespace(ns)` | 3 (components) | 点号路径，返回纯 `dict` |
| `Config.to_dict()` | 6, 13 | 返回纯 `dict` 深拷贝 |
| `Config.<attr>` 属性访问 | 6, 8 | 冻结后只读 |
| `validate_schema(params, schema)` | 3 (components) | 五项校验，失败抛 `ConfigConflictError` |
| `ConfigConflictError` | 3, 6, 8 | 异常携带 `errors: list[str]` |
| `FrozenConfigError` | 3, 6, 8 | 异常携带 `key` + `location` |
| `SearchSpace` | 8 (sweep) | dataclass, `expression: str` |
