<div align="center">

<img src="https://wgjgdnuofimjlhgwwicy.supabase.co/storage/v1/object/public/public-assets/images/1783878624995_neko5.jpg" width="100%" alt="Banner"/>

<h1>NameFrame 深度学习框架</h1>

![authors](https://img.shields.io/badge/authors-Aloudname-162b3e?style=flat-square)
![status](https://img.shields.io/badge/status-Developing-ffdd00?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-blue?style=flat-square)
![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)

</div>

---

> *注意！此项目重构，仍在开发中！*

## 前言

旧的框架并不好用。使用过程中发现其接口过于开放非常难用、组件间通信方式乱、初始化和管理方法完全暴露不好控制等等。这已经不是一般的问题了，必须要出重拳！艰难地品尝自己烹饪的东西后，决定回锅重炒，根据上一版应用过程中的问题，结合一些毕设过程中的实践心得，提出底层的设计思想，在此基础上重新开发。这次，我一定要成功口牙！嘻嘻！♪(^∇^*)

## API 文档

项目每层都提供一些 API，以及相关语法和应用例。

[本地 API 文档](docs/index.md) · [ReadTheDocs API 文档](https://nameframe.readthedocs.io/zh-cn/latest/) · [维护与兼容](docs/maintenance.md) · [版本变化](CHANGELOG.md)

---

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

## 参与开发

在源码根目录安装开发依赖：

```bash
python -m pip install -e ".[dev]"
```

相关检查、文档依赖与发行物验收统一见[维护说明](docs/maintenance.md)，普通使用无需安装这些开发工具。

MIT License，作者 Aloudname。
