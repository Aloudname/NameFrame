# 数据管线 API

> `nameframe.data` 负责逐样本变换、批次拼接和 DataLoader 构造。设备迁移、模型调用及训练循环由用户安排。

## 一、从样本到批次

[数据集](components.md) 负责读取原始 Batch，DataPipeline 负责执行变换，再将多个样本拼接成批次。完整顺序是：

原始 Batch → dataset.transforms(mode) 中的逐样本变换 → collate → 批次 Batch。

下面使用一个不继承 NameFrame 基类的数据集，示例可以独立运行：

```python
import torch

from nameframe.components import Batch
from nameframe.data import DataPipeline


class Samples:
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return Batch(data=torch.tensor([float(index)]), meta={"id": index})

    def transforms(self, mode="train"):
        return []


loader = DataPipeline(Samples(), batch_size=4, mode="infer").get_loader()
for _ in range(2):
    ids = [item for batch in loader for item in batch.meta["id"]]
    assert ids == list(range(8))
```

## 二、模式与 loader 的使用

mode 支持 train 和 infer。train 默认打乱样本，infer 默认保持顺序，也可用 shuffle 显式指定。启用哪些变换由 `dataset.transforms(mode)` 决定，模式不会自动划分训练集和验证集。

每次调用 `get_loader()`，都会创建独立的数据视图和变换链，不修改原数据集的模式。因此，同一个数据集可以分别创建训练和推理 loader。有状态变换的创建方法见 [Dataset](components.md#22)。

跨轮次使用时，像上例一样保留并复用返回的 loader；启用 persistent_workers 时尤其如此。直接迭代 DataPipeline 也可以，但每次迭代都会重新构造 loader 和变换链。

## 三、变换与拼接

### 3.1 逐样本变换

变换接收并返回 Batch，可以是普通函数，也可以是可调用对象；Transform 基类是可选的。原本处理 Tensor 或 PIL 图像的变换，需要包装成 Batch 变换，并负责同步更新标签和元数据。

管线在读取原始样本后、每次变换后检查 Batch 及其字段类型。错误信息包含数据集、样本索引和变换位置。批量读取还会检查返回值是序列，且数量与请求索引一致。

### 3.2 默认拼接

默认 collate 将 list[Batch] 拼成一个 Batch：data 和非空 target 分别通过 torch.stack 沿第 0 维堆叠，因此对应字段的形状和设备必须一致。target 全为 None 时，批次的 target 也是 None；只有部分样本缺少标签则报错。

meta 按键收集为逐样本列表，键的顺序取第一次出现的顺序。某个样本缺少该键，或值本身是 None，列表中都记作 None。

### 3.3 自定义拼接

变长序列等输入可传入 `collate_fn`，自行补齐或整理。DataPipeline 要求它返回 Batch，但当前不会再次检查拼接结果，需要在自定义函数中保证返回格式。

设 `batch_size=None` 可关闭自动分批，默认检查后原样返回单个 Batch；此时自定义 collate 收到的也是单个样本，不能同时设置 drop_last=True 或 batch_sampler。

## 四、加载参数

DataPipeline 接收 dataset、batch_size、mode、shuffle 和 num_workers，其他加载参数传给 DataLoaderFactory。常用设置如下：

| 参数 | 说明 |
|------|------|
| `batch_size=1` | 正整数；None 表示不自动分批 |
| `sampler=None` | 自定义采样顺序，指定后忽略 shuffle |
| `batch_sampler=None` | 同时管理采样和分批；须保持 batch_size=1、sampler=None、drop_last=False |
| `drop_last=False` | 是否丢弃不足一批的末尾样本 |
| `num_workers=0` | 非负整数，0 表示在主进程读取 |
| `prefetch_factor=None` | 多 worker 时默认 2；显式指定须为正整数且 num_workers>0 |
| `persistent_workers=False` | 保留 worker 供后续轮次复用，要求 num_workers>0 |
| `pin_memory=False` | 由 PyTorch 调用 Batch.pin_memory |
| `generator` / `worker_init_fn` | 传给 PyTorch，用于控制随机源和 worker 初始化 |

num_workers=0 时，不能指定 multiprocessing_context 或非零 timeout。使用多进程时，变换和初始化函数应能序列化；spawn 模式下应使用模块级定义，并以 `if __name__ == "__main__":` 保护程序入口。

## 五、只使用加载工厂

已有代码已完成变换时，可以从 `nameframe.data` 导入 DataLoaderFactory，调用 `create(dataset, ...)`。它只构造 loader 和处理拼接，不调用 dataset.transforms。

直接使用工厂时，自定义 collate 可以接收其他样本类型、返回其他结构，格式由用户自行约定。需要统一的 Batch 检查和按模式变换时，使用 DataPipeline。两种入口都不自动迁移设备，应在主进程取得批次后执行 `batch.to(device)`。
