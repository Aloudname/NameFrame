# NameFrame 使用文档

NameFrame 是面向个人研究项目的轻量 PyTorch 内核，提供可独立使用的配置、组件注册、Batch 和数据管线。用户保留普通 nn.Module、训练循环和实验状态的控制权。

## 从最小链路开始

取得源码并进入项目根目录，在已准备 PyTorch 的虚拟环境中执行：

```bash
python -m pip install .
python -m examples.minimal_train
```

`examples/minimal_train.py` 是完整、离线、CPU 可运行的主示例，随源码包提供；wheel 仅安装 nameframe。API 页注明“源码目录运行”的片段复用这个示例，其他可运行 Python 示例各自包含所需定义。

数据链路为：原始 Batch → 模式对应的逐样本变换 → collate → 批次 Batch → 调用方迁移设备 → 模型 → 损失与指标。

先使用 Batch 和所需数据处理，再按需接入配置或注册机制。Dataset 基类与 Metric 基类均是可选便利入口；自定义实现遵循对应契约即可。训练编排和状态持久化由调用方负责。

## 查阅入口

- [组件](api/components.md)：数据契约、模型/损失约定和受支持扩展点。
- [数据管线](api/data.md)：执行顺序、loader 生命周期、拼接和 worker。
- [配置](api/config.md)：可选的 YAML、冻结、校验与构造。
- [注册与工具](api/base.md)：可选的名称查找、发现、协议和随机源工具。
- [维护与兼容](maintenance.md)：契约边界、环境验证范围、开发与验收命令。

当前数据管线支持 map-style 数据集、单 Tensor 输入与 Tensor/None 标签。变长输入由自定义 collate 处理，具体模态语义由任务适配代码定义。
