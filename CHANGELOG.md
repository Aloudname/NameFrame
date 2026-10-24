# 版本变化


# 0.1.0.dev1：内核精简

* [X] 2026-10-24

### 1. 更新内容


- 我们不要算子辣！移除空 `engine/` 算子目录、`ops/plugin` 全局注册入口、搜索空间对象和 `Config.resolve`。
- 移除仅作协议转导出的 `components.model/loss`，以及 `BatchProtocol`；协议统一使用具体 `Batch`，配置元数据不再是计算协议必需项。
- Dataset 变换列表按实例隔离；有状态增强使用每个 loader 的变换工厂。
- fingerprint 采用显式 version，取消自动读文件和源码 /repr 推断。
- 保留 YAML/include/冻结、schema、注册与发现、Batch、数据管线及实际行为回归。
- 删除历史阶段设计和重复路线文档，公开边界、迁移和验收集中到维护说明。

从 dev0 迁移：删除 `Config.resolve()`；把 `BatchProtocol` 类型注解改为 `Batch`；模型/损失协议从 `components` 或 `utils` 导入；自定义 `Registry` 替代已删除的占位类别；为需要 fingerprint 的数据集声明 version 或覆写方法。有状态 `_transforms: Callable`移到`_augmentation_transforms()` 工厂。

### 2. 检查范围

| 检查                          | 结果                                                     |
| ----------------------------- | -------------------------------------------------------- |
| 源码完整回归                  | 239 通过，含真实双 worker spawn，无跳过                  |
| 源码包重建后隔离安装          | 239 通过，公开消费示例和最小训练通过；导入均来自安装目录 |
| 类型、Ruff、格式、差异空白    | 通过；消费样例包含有效调用和应拒绝的错误类型             |
| 文档、wheel、源码包及模块清单 | 通过，已删除模块不进入产物                               |
| CPU 最小训练                  | loss 0.8346 → 0.1943，accuracy 1.0                      |
| Python 3.10 语法              | 42 个源码/示例/测试/工具文件通过解析，不是 3.10 运行验证 |

环境为 Python 3.11.15、PyTorch 2.5.1+cu121、NumPy 2.4.6，计算使用 CPU。测试未纳入远端 CI 与 GPU。产出 `baseline.json` 与 SHA-256 清单。
