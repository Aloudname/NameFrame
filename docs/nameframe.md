# NameFrame Deep Learning Training Framework

## Background

Based on the `LoLA_hsViT` architecture, **NameFrame** is a standardized, modular deep learning training framework built by extracting and abstracting high-reuse patterns.
Its goal is to eliminate repetitive boilerplate when creating new deep learning projects, providing a consistent structure with well-decoupled components.

Key Functionalities:
- **Simple CLI**: `nameframe init my_project` initializes a project;
- **Config-driven**: Supports `.yaml` configuration and CLI parameter overrides;
- **Module registration**: Manage modules via registry;
- **Low-level acceleration**: Support CUDA/Triton/C++ acceleration through decorators.

---

## Quick Start

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

## 1. Architecture

### 1.1 Directory Structure

```
NameFrame/
├── nameframe/                        # Framework components
│   ├── __init__.py
│   ├── cli/                          # CLI: nameframe init|run|list|ops
│   ├── config/                       # Configuration templates
│   ├── registry/                     # Component registry (bottom layer)
│   ├── model/                        # BaseModel (ABC) + factory methods
│   ├── dataset/                      # BaseDataset (ABC) + data transforms + DataLoader factory methods
│   ├── loss/                         # BaseLoss (ABC)
│   ├── optim/                        # Optimizer + scheduler factory methods
│   ├── metrics/                      # BaseMetric (ABC) + built-in metrics
│   ├── pipeline/                     # Pipeline, Trainer, Analyzer, Visualizer, Monitor
│   ├── ops/                          # Accelerated implementations
│   │   ├── decorators.py             # Decorator definitions
│   │   ├── registry.py               # OP_REGISTRY + NATIVE_REGISTRY
│   │   ├── backends/                 # CUDA/Cython/cpp/triton compilation
│   │   ├── builtins/                 # Accelerated function definitions
│   │   ├── csrc/                     # Source files (.cu, .cpp)
│   │   └── verify.py                 # Verify source files and fallback
│   ├── utils/                        # Utilities (logging, checkpoint, seed, device)
│   └── run.py                        # User entry point
│
├── template/                         # Project skeleton for `nameframe init`
│   ├── config/config.yaml            # Project config (extends framework defaults)
│   ├── model/my_model.py             # User model
│   ├── dataset/preprocess.py         # Preprocessing logic
│   ├── dataset/dataset.py            # User dataset
│   ├── src/ops/                      # Project-specific decorator ops
│   ├── run.py                        # User entry point
│   └── .gitignore
│
├── setup.py / pyproject.toml
├── requirements.txt
└── README.md
```

### 1.2 Dependency Graph

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
                         │  Pipeline(core) │  <- Orchestrator
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

`ops/` is an independent module with no dependencies from other modules. Any module can import and call decorators directly.

---

## 2. Modules

### 2.1 Registry `nameframe/registry/`

Foundation layer. Provides a named registry for each pluggable component type.
**" \* "** denotes optional components.

| Symbol | Description |
|:------:|:------:|
| `Registry(name)` | Named string key -> callable mapping, supports `register()`, `get()`, `list()`, `__contains__` |
| `MODEL_REGISTRY` | Global model registry |
| `DATASET_REGISTRY` | Global dataset registry |
| `LOSS_REGISTRY` | Global loss registry |
| `METRIC_REGISTRY` | Global metric registry |
| `OPTIMIZER_REGISTRY` | Global optimizer registry* |
| `SCHEDULER_REGISTRY` | Global scheduler registry* |

### 2.2 Configuration `nameframe/config/`

| Function | Description |
|:------:|:------:|
| `load_config(path)` -> `Munch` | Load `.yaml`, resolve `${ENV_VAR}`, convert to attribute-accessible Munch |
| `merge_args(config, overrides)` -> `Munch` | Merge dotted-key overrides (e.g. `--set train.lr=0.001`) |
| `_deep_update(dst, src)` | Recursive dict merge |
| `_set_nested(container, dotted_key, value)` | Set value at dotted path, creating intermediate nodes as needed |

**Configuration layering**:

|  1. Framework defaults (`nameframe/config/template.yaml`)
|  2. Project config overrides (`config/config.yaml`)
v  3. CLI overrides (`--set key=value`)

### 2.3 Model Layer `nameframe/model/`

**`base.py` abstract interface:**

| Member | Description |
|:------:|:------:|
| `BaseModel(nn.Module)` | Abstract base class; new models must inherit this |
| `forward(x)` -> `Tensor[B,num_classes,H,W]` | Abstract method |
| `forward_features(x)` | Optional, returns encoder feature maps |
| `get_aux_outputs()` -> `dict` | Optional, auxiliary outputs for multi-loss scenarios |
| `forward_with_aux(x)` -> `(logits, aux_dict)` | Optional, wraps forward + get_aux_outputs |

**`builder.py` factory:**
- `build_model(config)` -> `nn.Module`: looks up `config.model.name` in `MODEL_REGISTRY`, instantiates with `config.model`.

**Extension:** Use `@MODEL_REGISTRY.register("my_model")` on a `BaseModel` subclass.

### 2.4 Dataset Layer `nameframe/dataset/`

**`base.py` abstract interface:**

| Member | Description |
|:------:|:------:|
| `BaseDataset(Dataset)` | Abstract base class; new datasets must inherit this |
| `__getitem__(idx)` -> `(Tensor[C,H,W], Tensor[H,W])` | Returns image and target |
| `__len__()` -> `int` | Dataset size |
| `num_classes` | Property, number of classes |
| `class_names` | Property, list of class names |

**`transforms.py`:**
- `build_transforms(config)` builds a `Compose` pipeline from config
- `AugmentationPipeline` image augmentation

**`builder.py` factory:**
- `build_dataloader(config, split)` -> `DataLoader` instantiates dataset from registry, wraps as DataLoader
- `build_dataloaders(config)` -> `Dict[str, DataLoader]` builds `{"train": ..., "val": ..., "test": ...}`

### 2.5 Loss Layer `nameframe/loss/`

**`base.py` abstract interface:**

| Member | Description |
|:------:|:------:|
| `BaseLoss(nn.Module)` | Abstract base class; new losses must inherit this |
| `forward(logits, targets, **kwargs)` -> `Tensor` | Abstract method, returns scalar loss |
| `get_components()` -> `Dict[str, float]` | Optional, per-component loss values for logging |

**`builtins.py` built-in losses:**

| Class | Description |
|:----:|:------:|
| `CrossEntropyLoss` | Cross-entropy with class weights and label smoothing |
| `DiceLoss` | Soft Dice loss, configurable foreground-only mode |
| `FocalLoss` | Focal loss with α and γ parameters |

### 2.6 Optimizer & Scheduler `nameframe/optim/`

| Function | Description |
|:------:|:------:|
| `build_optimizer(model, config)` -> `Optimizer` | AdamW/SGD/Adam |
| `build_scheduler(optimizer, config, steps_per_epoch)` -> `LRScheduler` | cosine, step, poly, plateau, onecycle, warmup_cosine |

### 2.7 Metrics Layer `nameframe/metrics/`

**`base.py` abstract interface:**

| Member | Description |
|:------:|:------:|
| `BaseMetric` | Abstract base class, supports stateless or accumulated computation |
| `update(preds, targets)` | Accumulate batch data |
| `compute()` -> `Dict[str, float]` | Compute final metric values |
| `reset()` | Reset accumulated state |

**`builtins.py`:** `Accuracy`, `DiceScore`, `ConfusionMatrix`, `MetricCollection` multi-metric aggregator.

### 2.8 Pipeline Layer `nameframe/pipeline/`

#### Pipeline Core `core.py`

Orchestrates the pipeline, depending on interfaces rather than concrete implementations.

```python
class Pipeline:
    def __init__(self, config)           # Receives config, initializes sub-modules
    def run() -> PipelineResult          # Full workflow
    def build_data() -> Dict[str, DataLoader]
    def build_model() -> nn.Module
    def build_loss() -> nn.Module
    def build_metrics() -> MetricCollection
    def train(model, loaders, loss_fn, metrics) -> TrainerResult
    def evaluate(model, loader, metrics) -> dict
    def export(model, format)            # ONNX / TorchScript
```

`PipelineResult`: contains `metrics`, `best_epoch`, `output_dir`, `checkpoint_path`, `onnx_path`.

#### Trainer `trainer.py`

Knows nothing about model architecture or data format. Only depends on `nn.Module`, `DataLoader`, `BaseLoss`, `MetricCollection`.
Features: AMP with configurable dtype, gradient accumulation, gradient clipping, EMA, automatic OOM recovery, early stopping, per-epoch callbacks, step/epoch LR scheduling.

| Method | Description |
|:------:|:------:|
| `__init__(model, config)` | Setup device, optimizer, scheduler, AMP, EMA, checkpoint manager |
| `fit(train_loader, val_loader, epochs)` -> `TrainerResult` | Main training loop with early stopping |
| `train_one_epoch(loader)` | Single epoch training |
| `validate(loader)` | Validation loop |
| `predict(loader)` | Inference with optional image retention |

#### Analyzer `analyzer.py`

Stateless metrics computer: `compute(preds, targets, probs)` -> `MetricsBundle`, `summarize(bundle)` -> formatted table, `per_class_report(bundle)` -> per-class P/R/F1.

#### Visualizer `visualize.py`

Publication-quality figures: training curves, confusion matrices, ROC/PR curves, segmentation examples, t-SNE/PCA feature distributions, LR schedule curves, gradient flow.

#### Monitor `monitor.py`

Background GPU/CPU resource monitor: `start()` / `stop()` -> `MonitorReport` (peak memory, avg utilization, timeline).

### 2.9 Utilities `nameframe/utils/`

| Module | Provides |
|:------:|:------:|
| `logging.py` | `tprint(*args)`, `setup_logger(name, log_file)` |
| `checkpoint.py` | `save_checkpoint()`, `load_checkpoint()`, `CheckpointManager(output_dir, keep_top_k)` |
| `seed.py` | `seed_everything(seed)` |
| `device.py` | `device(preferred=None)` -> `torch.device` |

### 2.10 CLI `nameframe/cli/main.py`

```bash
nameframe init my_project [--template <url>]    # Create new project
nameframe run [--config path] [--set k=v ...]   # Run training
nameframe list models|datasets|losses|metrics    # List registered components
nameframe ops build|status|verify|clean          # Manage accelerated ops
```

---

## 3. Accelerated Op Decorators `nameframe/ops/`

### 3.1 Decorator API

Provides decorators to accelerate certain ops/methods. When a decorator cannot resolve, the function body executes as the fallback.

| Decorator | Description |
|:------:|:------:|
| `@accelerated(name)` | Auto-dispatch, highly flexible |
| `@cuda_kernel(name, sources)` | CUDA |
| `@triton_kernel(name, config)` | Triton |
| `@cython_op(name, pyx_source)` | Cython |
| `@register_native(name, backend)` | Pre-compiled |

#### `@cuda_kernel(name, sources, build_dir, headers=None)`

Taking a custom average pooling method as an example, first implement the low-level CUDA operator in `.template/src/ops/csrc`:

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
    // Launch kernel, return result
    auto output = torch::zeros_like(x);
    ...
    return output;
}
```

Create `.template/src/ops/my_avg_pool.py`, define the method and decorate it:

```python
# .template/src/ops/my_avg_pool.py
import torch.nn.functional as F
from nameframe.ops import accelerated, cuda_kernel

# Fallback method + decorator = accelerated method
@cuda_kernel("my_avg_pool", sources=["src/csrc/my_ops.cu"])
def _native_my_avg_pool():
    """Placeholder function, replaced at runtime."""
    pass
```

For convenient use, register in `.template/src/ops/__init__.py`:

```python
from template.src.ops.my_avg_pool import my_avg_pool
__all__ = ["my_avg_pool"]
```

Usage:

```python
from template.src.ops import my_avg_pool
result = my_avg_pool(x, kernel_size=3)
```

#### `@triton_kernel(name, config)`

Taking a custom `fuse_scale + mask + softmax` block as an example, create `template/src/ops/my_block.py`, define the method and decorate it:

```python
# .template/src/ops/my_block.py
import torch, torch.nn.functional as F
import triton, triton.language as tl
from nameframe.ops import accelerated, triton_kernel

# Triton implementation of my_block
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

Similar to the CUDA decorator, register in `.template/src/ops/__init__.py`:

```python
from template.src.ops.my_block import my_block
__all__ = ["my_block"]
```

Usage:

```python
from template.src.ops import my_block
result = my_block(x)
```

#### `@cython_op(name, pyx_source)`

Taking custom non-maximum suppression as an example. `@cython_op` supports two modes: passing a `.pyx` file path and passing inline Cython source as a string.

**Approach 1: .pyx file path**
For more complex ops, organize them as standalone `.pyx` files.

```python
# .template/src/ops/csrc/nms.pyx
import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def my_nms(double[:, ::1] boxes, double[:] scores, double iou_threshold):
    # C loop implementation of NMS
    cdef list keep = []
    ...
    return keep
```

Create `my_nms.py` to implement the method:

```python
# .template/src/ops/my_nms.py
from nameframe.ops import accelerated, cython_op

@cython_op("my_nms", pyx_source="src/csrc/nms.pyx")
def _cython_my_nms():
    """Placeholder function, replaced at runtime with compiled Cython function."""
    pass
```

**Approach 2: Inline Cython source**
For simpler ops, create `my_nms.py` directly and pass the Cython source as a string parameter:

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

Same follow-up as above: register in `.template/src/ops/__init__.py`, then call directly.

#### `@register_native(name, backend)`

For cases where pre-compiled `.so` / `.pyd` already exist. Taking a matrix multiply-add op as an example, create `my_gemm.py`:

```python
# .template/src/ops/my_gemm.py
from nameframe.ops import accelerated, register_native
import ctypes

_lib = ctypes.CDLL("src/csrc/libmyops.so")

@register_native("my_gemm", backend="cuda")
def _cuda_my_gemm(a, b):
    pass
```

Same follow-up as above: register in `.template/src/ops/__init__.py`, then call directly.

#### `@accelerated(name, backends=None, config=None)`

When decorating a function with `@accelerated`, it uses a flexible low-level acceleration strategy, trying Triton, CUDA, Cython backends in order. If no low-level acceleration is available, it falls back to the Python method in the function body.

Create `.template/src/ops/my_avg_pool.py`, define the method and decorate it:

```python
@accelerated("my_avg_pool")
def my_avg_pool(x, kernel_size=3):
    """Fallback implementation"""
    return F.avg_pool2d(x, kernel_size, 
                stride=1, padding=kernel_size // 2)
```

Same follow-up as above: register in `.template/src/ops/__init__.py`, then call directly. The logic is:

```
@accelerated("my_avg_pool")
    1. Save the decorated function as ._fallback
    2. Register in OP_REGISTRY
    3. On first call, detect by priority:
        ├─ Triton available? -> replace with triton
        ├─ CUDA available?   -> JIT compile .cu, replace with CUDA
        ├─ C++ available?    -> replace with Cython
        └─ else              -> ._fallback
    4. On subsequent calls, use the previously resolved path
```

### 3.2 Accelerated Ops Configuration

```yaml
ops:
  build_strategy: "lazy"          # lazy: JIT on first call; eager: pre-build all
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

## 4. Extensibility

| Mechanism | How to Extend | Example |
|:------:|:------:|:------:|
| **Registry** | Add decorator to class/function | `@MODEL_REGISTRY.register("my_model")` |
| **Callbacks** | Add to `config.pipeline.callbacks` | `{name: "tensorboard", log_dir: "logs/"}` |
| **Accelerated ops** | Decorate function + optionally register native backend | `@accelerated("my_op")` |
| **Project template** | `nameframe init` creates an override-ready project skeleton | (>ω<) |
| **Config layering** | Edit `config/config.yaml` | (>ω<) |
| **CLI overrides** | `--set` flag | `--set train.epochs=200` |

---

## 5. Dependencies

See `requirements.txt`.

---

## 6. Features

| Step | Scope | Key Deliverables |
|:------:|:------:|:------:|
| 1 | **Package skeleton** | `setup.py`, `requirements.txt`, `nameframe/__init__.py` |
| 2 | **Foundation** | `utils/` (miscellaneous), `registry/` |
| 3 | **Config layer** | `config/loader.py`, `config/template.yaml` |
| 4 | **Abstract interfaces** | `model/base.py`, `dataset/base.py`, `loss/base.py`, `metrics/base.py`, all `builder.py`, `builtins.py` |
| 5 | **Pipeline layer** | `pipeline/` (core, trainer, analyzer, visualize, monitor) |
| 6 | **Entry points** | `nameframe/run.py`, `nameframe/cli/main.py` |
| 7 | **Project template** | `template/` with all stub files, end-to-end test `nameframe init` |
|  | **Ops layer** | `ops/decorators.py`, `ops/backends/`, `ops/builtins/`, `ops/verify.py` |

---
