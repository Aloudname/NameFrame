# NameFrame 深度学习训练框架

## 背景

基于 `LoLA_hsViT` 的架构，提取其高复用性的内容，进行抽象、优化，构建一个标准化、模块化的深度学习训练框架 **NameFrame**。
其目标是消除每次创建新深度学习项目时的重复样板代码，提供结构一致、组件解耦良好的框架。

Key Func：
- **CLI简单配置**：`nameframe init my_project` 初始化项目；
- **配置驱动**：支持 .yaml 配置和 CLI 参数覆盖；
- **模块注册**：通过注册表管理模块；
- **底层加速**：通过装饰器支持 CUDA/Triton/C++ 底层加速。

---

## 快速开始

```bash
# Install
cd nameframe && pip install -e .

# Create a new project
nameframe init my_project
cd my_project

# Edit config/config.yaml with your model/dataset settings,
# then implement model/my_model.py and dataset/dataset.py

# Run training
nameframe run
```

## 1. 架构

### 1.1 目录

```
NameFrame/
├── nameframe/                        # 框架组件
│   ├── __init__.py
│   ├── cli/                          # CLI: nameframe init|run|list|ops
│   ├── config/                       # 配置模板
│   ├── registry/                     # 组件注册，底层
│   ├── model/                        # BaseModel（抽象基类）+ 工厂方法
│   ├── dataset/                      # BaseDataset（抽象基类）+ 数据变换 + DataLoader 工厂方法
│   ├── loss/                         # BaseLoss（抽象基类）
│   ├── optim/                        # 优化器 + 调度器工厂方法
│   ├── metrics/                      # BaseMetric（抽象基类）+ 内置指标
│   ├── pipeline/                     # Pipeline、Trainer、Analyzer、Visualizer、Monitor
│   ├── ops/                          # 加速实现
│   │   ├── decorators.py             # 装饰器定义
│   │   ├── registry.py               # OP_REGISTRY + NATIVE_REGISTRY
│   │   ├── backends/                 # CUDA/Cython/cpp/triton 编译
│   │   ├── builtins/                 # 加速函数定义
│   │   ├── csrc/                     # 源文件（.cu、.cpp、.pyx）
│   │   └── verify.py                 # 验证源文件和回退
│   ├── utils/                        # 杂活（日志、检查点、随机种子、设备）
│   └── run.py                        # 用户入口
│
├── template/                         # `nameframe init` 使用的项目骨架
│   ├── config/config.yaml            # 项目配置（扩展框架默认值）
│   ├── model/my_model.py             # 用户模型
│   ├── dataset/preprocess.py         # 预处理逻辑
│   ├── dataset/dataset.py            # 用户数据集
│   ├── src/ops/                      # 项目特定的装饰器算子
│   ├── run.py                        # 用户入口
│   └── .gitignore
│
├── setup.py / pyproject.toml
├── requirements.txt
└── README.md
```

### 1.2 依赖关系

```
                         ┌─────────────────┐
                         │    run.py/CLI   │
                         └────────┬────────┘
                                  │ load_config, merge_args
                         ┌────────▼────────┐
                         │  config/loader  │
                         └────────┬────────┘
                                  │ Munch config
                         ┌────────▼────────┐
                         │  Pipeline(core) │  <- 编排器
                         └──┬───┬───┬───┬──┘
                            │   │   │   │
             ┌──────────────┘   │   │   └──────────────┐
             ▼                  ▼   ▼                  ▼
    ┌────────────┐   ┌──────────────┐   ┌──────────┬───────────┐
    │ dataset/   │   │  Trainer     │   │ Analyzer │ Visualizer│
    │ builder    │   └──┬───┬───┬───┘   │ Monitor  │           │
    └─────┬──────┘      │   │   │       └──────────┴───────────┘
          │             │   │   │
          │      ┌──────┘   │   └──────────┐
          │      ▼          ▼              ▼
          │  ┌────────┐ ┌────────┐  ┌──────────────┐
          │  │ model/ │ │ loss/  │  │ optim/       │
          │  │builder │ │ base   │  │ builder      │
          │  └───┬────┘ └───┬────┘  └──────┬───────┘
          │      │          │               │
          │      ▼          ▼               ▼
          │  ┌──────────┐ ┌──────────┐ ┌──────────────┐
          │  │BaseModel │ │BaseLoss  │ │AdamW/SGD +   │
          │  └──────────┘ └──────────┘ │Scheduler     │
          │                            └──────────────┘
          ▼
    ┌────────────┐
    │BaseDataset │
    └────────────┘
```

`ops/` 是独立模块，无任何模块对其依赖。任何模块都可以直接导入并调用装饰器。

---

## 2. 各模块

### 2.1 注册 `nameframe/registry/`

基础层，为每种可插拔组件类型提供具名注册表。
**“ * ”** 表示非必要组件。

| 符号 | 描述 |
|:------:|:------:|
| `Registry(name)` | 具名字符串键 -> 可调用对象映射，支持 `register()`、`get()`、`list()`、`__contains__` |
| `MODEL_REGISTRY` | 全局模型注册表 |
| `DATASET_REGISTRY` | 全局数据集注册表 |
| `LOSS_REGISTRY` | 全局损失函数注册表 |
| `METRIC_REGISTRY` | 全局指标注册表 |
| `OPTIMIZER_REGISTRY` | 全局优化器注册表* |
| `SCHEDULER_REGISTRY` | 全局调度器注册表* |

### 2.2 配置 `nameframe/config/`

| 函数 | 描述 |
|:------:|:------:|
| `load_config(path)` -> `Munch` | 加载 `.yaml`，解析 `${ENV_VAR}`，转换为属性可访问的 Munch |
| `merge_args(config, overrides)` -> `Munch` | 合并点号键覆盖（如 `--set train.lr=0.001`） |
| `_deep_update(dst, src)` | 递归字典合并 |
| `_set_nested(container, dotted_key, value)` | 在点号路径设值，按需创建中间节点 |

**配置分层**：

|  1. 框架默认值（`nameframe/config/template.yaml`）
|  2. 项目配置覆盖（`config/config.yaml`）
v  3. CLI 覆盖（`--set key=value`）

### 2.3 模型层 `nameframe/model/`

**`base.py` 抽象接口：**

| 成员 | 描述 |
|:------:|:------:|
| `BaseModel(nn.Module)` | 抽象基类，新定义模型应继承之 |
| `forward(x)` -> `Tensor[B,num_classes,H,W]` | 抽象方法 |
| `forward_features(x)` | 可选，返回编码器的特征图 |
| `get_aux_outputs()` -> `dict` | 可选，多损失时用于副损失的辅助输出 |
| `forward_with_aux(x)` -> `(logits, aux_dict)` | 可选，封装 forward + get_aux_outputs |

**`builder.py`工厂函数：**
- `build_model(config)` -> `nn.Module`：在 `MODEL_REGISTRY` 中查找 `config.model.name`，使用 `config.model` 实例化。

**扩展：** 在 `BaseModel` 子类上使用 `@MODEL_REGISTRY.register("my_model")`。

### 2.4 数据层 `nameframe/dataset/`

**`base.py` 抽象接口：**

| 成员 | 描述 |
|:------:|:------:|
| `BaseDataset(Dataset)` | 抽象基类，新定义数据集应继承之 |
| `__getitem__(idx)` -> `(Tensor[C,H,W], Tensor[H,W])` | 返回图像和目标 |
| `__len__()` -> `int` | 数据集大小 |
| `num_classes` | 属性，类别数 |
| `class_names` | 属性，类别名称列表 |

**`transforms.py`：**
- `build_transforms(config)` 从配置构建 `Compose` 管线
- `AugmentationPipeline` 图像增强

**`builder.py` 工厂函数：**
- `build_dataloader(config, split)` -> `DataLoader` 从注册表实例化数据集，封装为 DataLoader
- `build_dataloaders(config)` -> `Dict[str, DataLoader]` 构建 `{"train": ..., "val": ..., "test": ...}`

### 2.5 损失层 `nameframe/loss/`

**`base.py` 抽象接口：**

| 成员 | 描述 |
|:------:|:------:|
| `BaseLoss(nn.Module)` | 抽象基类，新定义损失应继承之 |
| `forward(logits, targets, **kwargs)` -> `Tensor` | 抽象方法，返回标量损失 |
| `get_components()` -> `Dict[str, float]` | 可选，各分量损失值，用于日志记录 |

**`builtins.py` 内置损失：**

| 类 | 描述 |
|:----:|:------:|
| `CrossEntropyLoss` | 带类别权重和标签平滑的交叉熵 |
| `DiceLoss` | 软 Dice 损失，可配置仅前景模式 |
| `FocalLoss` | 带 α 和 γ 参数的 Focal Loss |

### 2.6 优化器与调度器 `nameframe/optim/`

| 函数 | 描述 |
|:------:|:------:|
| `build_optimizer(model, config)` -> `Optimizer` | AdamW/SGD/Adam |
| `build_scheduler(optimizer, config, steps_per_epoch)` -> `LRScheduler` | cosine、step、poly、plateau、onecycle、warmup_cosine |

### 2.7 指标层 `nameframe/metrics/`

**`base.py` 抽象接口：**

| 成员 | 描述 |
|:------:|:------:|
| `BaseMetric` | 抽象基类，支持无状态或累积计算 |
| `update(preds, targets)` | 累积批次数据 |
| `compute()` -> `Dict[str, float]` | 计算最终指标值 |
| `reset()` | 重置累积状态 |

**`builtins.py`：** `Accuracy`、`DiceScore`、`ConfusionMatrix`、`MetricCollection` 多指标聚合器。

### 2.8 管线层 `nameframe/pipeline/`

#### 管线核心 `core.py`

编排管线，依赖接口而非具体实现。

```python
class Pipeline:
    def __init__(self, config)           # 接收配置，初始化各子模块
    def run() -> PipelineResult          # 完整工作流
    def build_data() -> Dict[str, DataLoader]
    def build_model() -> nn.Module
    def build_loss() -> nn.Module
    def build_metrics() -> MetricCollection
    def train(model, loaders, loss_fn, metrics) -> TrainerResult
    def evaluate(model, loader, metrics) -> dict
    def export(model, format)            # ONNX / TorchScript
```

`PipelineResult`：包含 `metrics`、`best_epoch`、`output_dir`、`checkpoint_path`、`onnx_path`。

#### 训练器 `trainer.py`

对模型架构和数据格式一无所知，仅依赖 `nn.Module`、`DataLoader`、`BaseLoss`、`MetricCollection`。
其具有可配置数据类型的 AMP、梯度累积、梯度裁剪、EMA、自动 OOM 恢复、早停、每轮回调、按步/按轮 LR 调度。

| 方法 | 描述 |
|:------:|:------:|
| `__init__(model, config)` | 设置设备、优化器、调度器、AMP、EMA、检查点管理器 |
| `fit(train_loader, val_loader, epochs)` -> `TrainerResult` | 主训练循环，支持早停 |
| `train_one_epoch(loader)` | 单轮训练 |
| `validate(loader)` | 验证循环 |
| `predict(loader)` | 推理，可选择性保留图像 |

#### 指标计算器 `analyzer.py`

无状态指标计算器：`compute(preds, targets, probs)` -> `MetricsBundle`、`summarize(bundle)` -> 格式化表格、`per_class_report(bundle)` -> 每类 P/R/F1。

#### 可视化集合 `visualize.py`

出版级图表：训练曲线、混淆矩阵、ROC/PR 曲线、分割示例、t-SNE/PCA 特征分布、LR 调度曲线、梯度流。

#### 资源监控 `monitor.py`

后台 GPU/CPU 资源监控器：`start()` / `stop()` -> `MonitorReport`（峰值内存、平均利用率、时间线）。

### 2.9 工具 `nameframe/utils/`

| 模块 | 提供方法 |
|:------:|:------:|
| `logging.py` | `tprint(*args)`、`setup_logger(name, log_file)` |
| `checkpoint.py` | `save_checkpoint()`、`load_checkpoint()`、`CheckpointManager(output_dir, keep_top_k)` |
| `seed.py` | `seed_everything(seed)` |
| `device.py` | `device(preferred=None)` -> `torch.device` |

### 2.10 CLI `nameframe/cli/main.py`

```bash
nameframe init my_project [--template <url>]    # 创建新项目
nameframe run [--config path] [--set k=v ...]   # 运行训练
nameframe list models|datasets|losses|metrics    # 列出已注册组件
nameframe ops build|status|verify|clean          # 管理加速算子
```

---

## 3. 加速算子装饰器 `nameframe/ops/`

### 3.1 decorator API

提供一些装饰器，加速某些算子/方法。装饰器失效时，执行函数体作为回退方法。

| 装饰器 | 描述 |
|:------:|:------:|
| `@accelerated(name)` | 能能能能，手段十分灵活 |
| `@cuda_kernel(name, sources)` | CUDA |
| `@triton_kernel(name, config)` | Triton |
| `@cython_op(name, pyx_source)` | Cython |
| `@register_native(name, backend)` | 预编译的 |

#### `@cuda_kernel(name, sources, build_dir, headers=None)`

以自定义的平均池化方法为例，先在 `.template/src/ops/csrc` 实现 `.cuda` 的底层算子：

```cpp
// src/csrc/my_avg_pool.cu
#include <torch/extension.h>

__global__ void my_avg_pool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int H, int W, int K
) {
    ...
}

torch::Tensor my_avg_pool_cuda(torch::Tensor
x, int kernel_size) {
    // 启动内核，返回结果
    auto output = torch::zeros_like(x);
    ...
    return output;
}
```

新建一 `.template/src/ops/my_avg_pool.py`，定义该方法并装饰之。

```python
# .template/src/ops/my_avg_pool.py
import torch.nn.functional as F
from nameframe.ops import accelerated, cuda_kernel

# 回退方法 + 装饰器 = 加速方法
@cuda_kernel("my_avg_pool", sources=["src/csrc/my_ops.cu"])
def _native_my_avg_pool():
    """占位函数，运行时被替换。"""
    pass
```

为方便调用，在 `.template/src/ops/__init__.py` 注册:

```python
from template.src.ops.my_avg_pool import my_avg_pool
__all__ = ["my_avg_pool"]
```

使用：

```python
from template.src.ops import my_avg_pool
result = my_avg_pool(x, kernel_size=3)
```

#### `@triton_kernel(name, config)`

以自定义的 `fuse_scale + mask + softmax` 块为例，新建一 `template/src/ops/my_block.py`，定义该方法并装饰之。

```python
# .template/src/ops/my_block.py
  import torch, torch.nn.functional as F
  import triton, triton.language as tl
  from nameframe.ops import accelerated, triton_kernel

# my_block方法的Triton实现
  @triton.jit
  def _my_block(
      x_ptr, y_ptr,
      N: tl.constexpr,
      BLOCK_SIZE: tl.constexpr,
  ):
      ...
      pass

  @triton_kernel("my_block", autotune_configs=[
      {"BLOCK_SIZE": 128},
      {"BLOCK_SIZE": 256},
      {"BLOCK_SIZE": 512},
  ], num_warps=4, num_stages=2)
  def _triton_my_block():
      pass
```

与 cuda 修饰器类似，为方便调用，在 `.template/src/ops/__init__.py` 注册:

```python
from template.src.ops.my_block import my_block
__all__ = ["my_block"]
```

使用：

```python
from template.src.ops import my_block
result = my_block(x)
```

#### `@cython_op(name, pyx_source)`

以自定义的非极大值抑制为例。`@cython_op` 支持传入 `.pyx` 文件路径和直接传入内联 Cython 源码字符串两种模式。

方式一：.pyx 文件路径
针对算子比较复杂的情况，将其单独组织成 `.pyx` 文件。

```python
# .template/src/ops/csrc/nms.pyx
import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def my_nms(double[:, ::1] boxes, double[:] scores, double iou_threshold):
    # C 循环实现 NMS
    cdef list keep = []
    ...
    return keep
```

新建一 `my_nms.py` 实现方法：

```python
# .template/src/ops/my_nms.py
from nameframe.ops import accelerated, cython_op

@cython_op("my_nms", pyx_source="src/csrc/nms.pyx")
def _cython_my_nms():
    """占位函数，运行时会替换为编译后的 Cython 函数。"""
    pass
```

方式二：内联 Cython 源码
算子比较简单的情况。直接新建一 `my_nms.py` 实现方法，将 Cython 字段做字符串参数传入：

```python
# .template/src/ops/my_nms.py
@cython_op("my_nms", pyx_source=r"""
    import cython
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def my_nms(double[:, ::1] boxes, double[:] scores, double iou_threshold):
        cdef int i, j
        cdef list keep = []
        # ...
        return keep
    """)
def _cython_my_nms():
    pass
```

后续流程同上，在 `.template/src/ops/__init__.py` 注册，使用时直接 call。

#### `@register_native(name, backend)`

适用于已有预编译 .so / .pyd 的场景。以矩阵乘加算子为例，新建一 `my_gemm.py` 实现方法：

```python
# .template/src/ops/my_gemm.py
from nameframe.ops import accelerated, register_native
import ctypes

_lib = ctypes.CDLL("src/csrc/libmyops.so")

@register_native("my_gemm", backend="cuda")
def _cuda_my_gemm(a, b):
    pass
```

后续流程同上，在 `.template/src/ops/__init__.py` 注册，使用时直接 call。

#### `@accelerated(name, backends=None, config=None)`

用 `@accelerated` 装饰函数时，采用灵活的底层加速方法，依次尝试搜索 Triton、CUDA、Cython 底层加速。若底层加速未实现，回退至函数体内的 Python 方法。

新建一 `.template/src/ops/my_avg_pool.py`，定义该方法并装饰之。

```python
@accelerated("my_avg_pool")
def my_avg_pool(x, kernel_size=3):
    """回退实现"""
    return F.avg_pool2d(x, kernel_size, 
                stride=1, padding=kernel_size // 2)
```

后续流程同上，在 `.template/src/ops/__init__.py` 注册，使用时直接 call。其逻辑是：
```
@accelerated("my_avg_pool")
    1. 将装饰的函数保存为 ._fallback
    2. 在 OP_REGISTRY 注册
    3. 首次调用时，按优先级检测：
        ├─ Triton 可用？ -> 替换为 triton
        ├─ CUDA 可用？   -> JIT 编译 .cu，替换为 CUDA
        ├─ C++ 可用？    -> 替换为 Cython
        └─ else         -> ._fallback
    4. 后续调用时，直接使用先前路径
```

### 3.2 加速算子相关配置

```yaml
ops:
  build_strategy: "lazy"          # lazy: 首次调用时 JIT；eager: 启动时全编译
  backend_priority: [triton, cuda, cython, python]
  verify_on_first_call: true
  verify_tolerance: 1.0e-4
  cache_dir: "~/.cache/nameframe/ops"
  cuda: {arch: ["8.0", "9.0"], jit_opt_level: "-O3"}
  triton: {num_warps: 4, num_stages: 2}
  cython: {language_level: 3, compiler_directives: {boundscheck: false, wraparound: false}}
  custom: {enabled: true, source_dir: "src/ops", native_dir: "src/csrc"}
```

---

## 4. 扩展方法

| 机制 | 扩展方法 | 例 |
|:------:|:------:|:------:|
| **注册表** | 对类/函数添加装饰器 | `@MODEL_REGISTRY.register("my_model")` |
| **回调** | `config.pipeline.callbacks` 中添加 | `{name: "tensorboard", log_dir: "logs/"}` |
| **加速算子** | 装饰函数 + 可选注册原生后端 | `@accelerated("my_op")` |
| **项目模板** | `nameframe init` 创建可覆盖的项目骨架 | (>ω<) |
| **配置分层** |  改 `config/config.yaml` | (>ω<) |
| **CLI 覆盖** | `--set` 参数 | `--set train.epochs=200` |

---

## 5. 项目依赖

见 `requirements.txt`。

---

## 6. 功能

| 步骤 | 范围 | 关键交付件 |
|:------:|:------:|:------:|
| 1 | **包骨架** | `setup.py`、`requirements.txt`、`nameframe/__init__.py` |
| 2 | **基础层** | `utils/`（杂项）、`registry/` |
| 3 | **配置层** | `config/loader.py`、`config/template.yaml` |
| 4 | **抽象接口** | `model/base.py`、`dataset/base.py`、`loss/base.py`、`metrics/base.py`、所有 `builder.py`、`builtins.py` |
| 5 | **管线层** | `pipeline/`（core、trainer、analyzer、visualize、monitor） |
| 6 | **入口点** | `nameframe/run.py`、`nameframe/cli/main.py` |
| 7 | **项目模板** | `template/` 包含所有桩文件，端到端测试 `nameframe init` |
|  | **算子层** | `ops/decorators.py`、`ops/backends/`、`ops/builtins/`、`ops/verify.py` |

---
