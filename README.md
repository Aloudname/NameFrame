# NameFrame

NameFrame 是面向个人研究项目的轻量 PyTorch 内核，提供可独立使用的配置、组件注册、Batch 和数据管线。模型继续使用普通 `nn.Module`，训练循环与实验状态由调用方管理。

最简意味着只组合实际需要的模块；稳定意味着契约明确、错误可定位、已有行为有回归验证；可拓展意味着通过受支持的接口适配任务。当前是开发版本，升级前请查看[版本变化](CHANGELOG.md)。

## 安装与运行

取得源码并进入项目根目录，在虚拟环境中按硬件准备 PyTorch 后执行：

```bash
python -m pip install .
python -m examples.minimal_train
```

主示例位于 `examples/minimal_train.py`，完全离线，在 CPU 上完成数据变换、训练和指标计算。示例随源码提供，wheel 仅安装 nameframe 包。Python 3.10+ / PyTorch 2.0+ 是安装边界，实际验证范围见[维护说明](docs/maintenance.md)。

## 内核边界

- 数据管线使用具体 `Batch`，逐样本变换接收并返回 Batch；模型返回 Tensor，损失约定返回标量 Tensor。
- 配置、注册表和 Dataset 基类均可按需采用，不要求为了使用一个模块接入整个库。
- 受支持的变换工厂为每个 loader 创建有状态变换，调用方负责避免修改共享 Tensor。
- 数据版本显式声明；内核不扫描文件或推测预处理状态。
- 训练编排、checkpoint、设备调度和分布式执行由调用方或其他工具负责，当前内核不提供这些能力。

[使用与 API 文档](docs/index.md) · [维护与兼容](docs/maintenance.md) · [版本变化](CHANGELOG.md)

## 参与开发

在源码根目录安装开发依赖：

```bash
python -m pip install -e ".[dev]"
```

相关检查、文档依赖与发行物验收统一见[维护说明](docs/maintenance.md)，普通使用无需安装这些开发工具。

MIT License，作者 Aloudname。
