# 维护与兼容

> 本页说明接口兼容、开发检查和发行流程。各模块的用法见 [API 文档](api/base.md)。

## 一、接口与兼容

NameFrame 保留配置、注册、Batch 和数据管线，训练循环、checkpoint 和设备调度由用户管理。各模块可以独立使用，改动时应保持已有调用方式清晰、可验证。

API 文档列出的入口是对外接口。其中，Dataset 的 `_augmentation_transforms()`、`_base_transforms()`、`transforms()` 和 `fingerprint()` 都允许覆写，前两个方法虽带下划线，也属于扩展接口。其他未公开的内部属性、辅助类和反射操作不保证兼容。

项目目前处于开发期，接口变化和迁移方法记录在仓库根目录的 `CHANGELOG.md`。修改公开行为时，应同步说明使用方式的变化，并补充对应回归检查。

## 二、运行环境

安装要求为 `Python 3.10+、PyTorch 2.0+`。这表示允许安装的版本范围，实际兼容情况仍应查看对应环境的运行记录，特别是 PyTorch 与 NumPy 的搭配。

| 环境 | Python / PyTorch / NumPy | 验证情况 |
|------|--------------------------|----------|
| 本地候选 | 3.11.15 / 2.5.1+cu121 / 2.4.6 | 已有 CPU 运行记录，详见 CHANGELOG |
| 最低版本 CI | 3.10 / 2.0.1 / 1.26.4 | 已配置，结果以远端运行为准 |
| 主要版本 CI | 3.11 / 2.5.1 CPU wheel / 2.4.6 | 已配置，结果以远端运行为准 |
| GPU 与其他平台 | — | 暂无验证记录 |

本地测试时，使用了 `PyTorch 2.5.1+cu121`，但计算在 CPU 上完成，尚未完成 GPU 验证。历史记录只对应当次源码和产物，后续修改需要重新检查。

## 三、开发检查

在源码根目录创建虚拟环境，并安装开发工具和文档依赖。`.venv` 已存在时跳过 `uv venv`：

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
uv pip install -r requirements-docs.txt
```

普通源码检出可以检查包的格式、构建文档和打包：

```bash
uv run --no-sync ruff check nameframe
uv run --no-sync ruff format --check nameframe
uv run --no-sync mkdocs build --strict
uv build
```

完整回归还需要维护者本地的 `tests/`、`tools/` 和 `examples/`。这三个目录目前不纳入版本控制，公开检出时需要另行准备；保留它们的开发环境可继续执行：

```bash
uv run --no-sync ruff check tests examples tools
uv run --no-sync ruff format --check tests examples tools
uv run --no-sync python -m mypy
uv run --no-sync python -m pytest
uv run --no-sync python -m examples.minimal_train
```

类型检查以 `tests/typing/public_api.py` 模拟公开接口的使用，同时保留应被拒绝的错误类型。它覆盖常见调用，不能替代运行测试。多进程测试若因环境限制跳过，应另外记录实际 worker 的运行结果。

远端 CI 只检查公开内容：`nameframe` 的 ruff 检查与格式检查、文档构建，以及 wheel 从源码包重建后的安装冒烟。完整回归留在本地执行。

## 四、发行检查与记录

`uv build` 生成 wheel 和源码包。准备发行时，还应从源码包重建 wheel（`uv build --wheel --out-dir <输出目录> dist/nameframe-*.tar.gz`），在仓库外安装，并将本地测试和示例复制到外部目录运行。先确认 `nameframe.__file__` 指向安装目录，避免误用源码。记录时区分只隔离安装了 NameFrame，还是重新安装了全部依赖。

检查记录应包含版本、源码标识、产物校验值、实际命令、环境和结果。有未提交修改时，Git HEAD 不能完整标识当次源码，还需保存对应差异或快照。

`dist/` 不进入版本控制。已有 baseline.json 和 SHA-256 清单是当次的本地归档，构建命令不会自动生成测试报告。源码或产物变化后，需要重新检查，不能沿用旧记录。
