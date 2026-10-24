# 配置系统 API

> `nameframe.config` 负责配置的加载、读取、冻结和校验。需要集中管理参数时，可将配置交给 `from_config` 构造组件；也可以直接用 Python 构造组件。

## 一、加载配置

配置可以来自普通字典，也可以来自 YAML 文件。两种入口最终都得到 Config 对象，后续读取和冻结的方法相同。

YAML 中使用 `!include` 引用其他文件，路径相对于当前文件。例如，先定义实验配置：

```yaml
# exp.yaml
model: !include model/linear.yaml
training:
  lr: 0.01
```

再单独保存模型参数：

```yaml
# model/linear.yaml
in_features: 4
out_features: 2
```

调用 `Config.from_yaml("exp.yaml")` 即可加载。临时调整参数时，传入 `overrides={"training": {"lr": 0.005}}`；映射递归合并，其他值整体替换，列表不逐项合并。覆盖不会改写原文件。

YAML 根节点必须是映射，空文档视为空映射。配置值支持字符串、bool、int、float、None、列表和字符串键映射；日期应加引号，避免被 YAML 解析为日期对象。加载使用 SafeLoader，不构造 Python 对象。

重复显式键会报错，并指出文件和位置；YAML 的 `<<` 合并仍允许显式字段覆盖继承值。`!include` 可以重复引用同一文件，但不能循环引用；错误会保留引用来源。循环容器同样会被拒绝，普通别名则复制为独立容器。

## 二、读取与冻结

加载后，可用属性或方括号访问字段。准备好参数后调用 `freeze()`，再交给组件使用：

```python
from nameframe.config import Config

config = Config({"training": {"lr": 0.01, "milestones": [10, 20]}})
assert config.training.lr == config["training"]["lr"]

config.freeze()
params = config.get_namespace("training")
params["milestones"].append(30)

assert config.is_frozen
assert list(config.training.milestones) == [10, 20]
```

`get_namespace("training.optimizer")` 提取指定路径下的普通 dict；路径缺失时抛 AttributeError，目标不是映射时抛 TypeError。`to_dict()` 则导出整棵配置树。两者都返回独立容器，修改导出值不会影响配置，传入 Config 的原始容器也不与它共享。

`freeze()` 返回自身，可以重复调用。冻结后，通过公开接口修改或删除字段、增删和重排列表，都会抛 FrozenConfigError；冻结前取得的内部引用也受保护。异常带有字段路径和调用位置，方便定位修改来源。私有实现和反射操作不属于冻结接口。

与方法重名的配置键应使用方括号，例如 `config["to_dict"]`。配置只保存具体参数，组件构造不接受 `${search: ...}` 搜索表达式。

## 三、校验参数

读入配置，只说明文件格式正确，还需要检查参数是否适合组件。组件可以用 config_schema 声明字段规则，再调用 validate_schema 校验：

```python
from nameframe.config import validate_schema
from nameframe.utils import FieldSchema

schema: dict[str, FieldSchema] = {
    "width": {"type": int, "default": 4, "range": (1, 1024)},
    "activation": {"type": str, "choices": ["relu", "gelu"]},
}
params = {"activation": "relu"}
validate_schema(params, schema)

assert params == {"activation": "relu"}
```

校验不会修改参数，也不会填入默认值。上例允许省略 width，但若将其设为字符串或超出范围，便会抛出 ConfigConflictError；该异常的 `errors` 属性保存错误列表。

每个字段用 FieldSchema 描述，可声明以下规则：

| 规则 | 说明 |
|------|------|
| `type` | Python 类型，按 isinstance 检查，不自动转换 |
| `default` | 缺省值；未声明时，该字段必填 |
| `help` | 字符串说明 |
| `choices` | 允许值的 list；None 表示不限 |
| `range` | 数值闭区间；None 表示不限 |
| `nullable` | 是否允许 None，默认 False |

未知字段、非法规则和值都会报错。`validate_schema_definition(schema)` 可单独检查规则；默认值也必须合法，即使本次传入了其他值。int/float 字段不接受 bool，直接校验的 float 值必须有限；range 允许无穷边界，但不接受 NaN 或倒置区间。

校验按字段进行。如果某字段的类型是 dict 或 list，只检查容器本身，不检查内部元素的类型、范围和有限性；这些嵌套内容需要另外校验。配置可以递归冻结，并不意味着 schema 也会递归校验。

## 四、从配置构造组件

参数准备好后，可用 `nameframe.components.from_config` 提取配置并调用目标类或工厂：

```python
from torch import nn

from nameframe.components import from_config
from nameframe.config import Config

config = Config(
    {
        "model": {"name": "linear", "in_features": 4, "out_features": 2},
    }
).freeze()
classifier = from_config(nn.Linear, config, "model")

assert classifier.in_features == 4
```

`from_config(cls, config, namespace)` 先检查配置已冻结，再提取 namespace 下的参数、移除用于名称查找的 name。有 config_schema 时，先校验参数，再复制缺省值，最后调用 `cls(**params)`。未声明 schema 的第三方类可直接使用；显式空 schema 则不允许额外构造参数。

校验错误会带上 namespace，构造函数内部的异常原样传出。旧 schema 中的 name 不作为构造参数，其规则定义仍会检查；业务参数应另取名称。

复制的默认值不会写回 Config，记录实际构造参数时需要一并保存它们。注册表只负责查找 cls，可按[基础 API](base.md) 接入，也可像上例一样直接传入类。
