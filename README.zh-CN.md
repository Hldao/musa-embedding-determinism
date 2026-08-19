# torch_musa Embedding Backward 确定性修复

这是一个非官方的最小复现与修复建议，用于解决 torch_musa
`aten::embedding_dense_backward` 在特定条件下不能稳定复现的问题。

项目由 **WKK AI R&D** 维护，与摩尔线程及 PyTorch 项目不存在隶属关系，也不代表其官方意见。

[English](README.md)

## 问题概述

在我们实测的 MTT S4000 环境中，embedding indices 存在重复 ID 时，即使输入、seed 和 PyTorch
确定性开关完全相同，BF16 embedding 梯度仍可能在不同新进程中产生不同结果。

公开的 torch_musa v1.3.0 实现会在输入较短且 `scale_grad_by_freq=False` 时选择 atomic 快路。
这条快路没有检查 PyTorch 的确定性设置。多个 GPU 线程向同一 embedding 行进行浮点 atomic
累加时，执行顺序变化可能改变最后的 bit pattern。

## 修复方式

torch_musa 已经有 sorted/segmented 实现，因此不需要引入新的梯度算法。补丁只增加一项路由条件：

```cpp
const bool use_atomic_fast_path =
    num_indices <= 3072 && !scale_grad_by_freq &&
    !globalContext().deterministicAlgorithms();
```

由此得到：

- 默认模式继续使用现有 atomic 快路，行为和性能不变；
- 用户显式调用 `torch.use_deterministic_algorithms(True)` 时，改走已有 segmented 路径；
- 不改 embedding forward，不影响普通前向推理；
- 不改变梯度的数学定义。

补丁同时在 sorted 路径中处理 `num_indices == 0`：直接返回全零梯度。这样开启
deterministic 后，空输入从 atomic 路由切到 sorted 路由时仍保持原有语义。
由于 v1.3 的 muDNN `Sort` 不接受空输入，兼容补丁会在调用它之前短路，同时保留
上游评审要求的 kernel 层保护；当前上游使用了不同的排序辅助逻辑，因此 PR 本身
只需要评审指定的 kernel 层保护。

## 实测环境与结果

- 设备：MTT S4000；
- PyTorch：2.2.0；
- 未修复 torch_musa wheel：1.3.0+81caf0a；
- 隔离验证构建：1.3.0+3d6a817；
- 输入：2,063 个 indices、157 个 unique ID、宽度 128 的 BF16 gradient。

`1.3.0+81caf0a` 是已安装厂商 wheel 自报的版本标识；对应源码 commit 在 torch_musa 公共仓库中
不可获得。

| 模式 | 新进程数 | 不同哈希数 | 结果 |
| --- | ---: | ---: | --- |
| 未修复 native | 4 | 4 | 非确定 |
| 未修复 + 请求 deterministic | 4 | 4 | 非确定 |
| 补丁后 native | 8 | 8 | 默认快路保持不变 |
| 补丁后 + 请求 deterministic | 8 | 1 | 确定 |
| 已有 segmented 路径规避 | 8 | 1 | 确定 |

补丁确定性路径与 segmented 规避路径的输出哈希完全相同。原始非空输入回归测试在原 wheel 上失败，
在隔离补丁构建上变为 `1 passed`；独立测试现已补充上游审查中指出的空 indices 边界。非空输入的
旧 v1.3 兼容构建会在该边界上报 `SortRun` 失败，更新后的构建两项回归测试均通过（`2 passed`）。
机器可读结果见 [`results/`](results/)。

## 复现

前提条件是系统中已有兼容的 MUSA 驱动、PyTorch、torch_musa 和可用 MUSA 设备。

```bash
python3 reproduce_embedding_backward.py --runs 8
```

脚本会用新子进程分别运行 native、请求 deterministic、segmented workaround 三种模式，并输出
JSON。在受影响且未修复的环境中，前两种模式可能出现多个哈希，workaround 只有一个；打补丁后，
deterministic 模式应只有一个哈希，并与 workaround 一致。

复现输入由数学函数生成，不包含模型、checkpoint、用户数据或任何私有训练方法。

## 应用补丁与测试

补丁按公开 torch_musa v1.3.0 源码布局制作：

```bash
cd /path/to/torch_musa
git apply /path/to/musa-embedding-determinism/patches/embedding-determinism.patch
```

请根据自己的 MUSA 运行环境使用官方构建方式。不要用实验 wheel 覆盖可工作的系统环境，建议使用
虚拟环境或隔离安装目录。

构建后可执行独立回归测试：

```bash
pytest -q proposed_test_embedding_determinism.py
```

该测试是 upstream 提案格式；正式合并时可由维护者放入现有 embedding 测试模块，并扩充 dtype、
shape 和设备覆盖。

## 性能微基准

```bash
python3 benchmark_embedding_backward.py --warmup 25 --rounds 200
```

在上述 synthetic shape 中，25 次预热、200 次设备同步计时的中位数为：

| 模式 | 中位延迟 | 相对 native |
| --- | ---: | ---: |
| Native atomic | 0.9701 ms | 1.000x |
| 补丁 deterministic | 1.0233 ms | 1.055x |
| Padding workaround | 1.0708 ms | 1.104x |

这只是单设备算子微基准，不代表完整模型训练吞吐。要评价其他架构或真实工作负载，应重新测量。

## 证据边界

- 结果只证明所列设备、软件版本和输入形状，不外推所有 MUSA 版本、卡型、dtype 或 shape。
- 验证构建基于公开 torch_musa v1.3.0、已安装运行时携带的 operator schema 和适用兼容补丁，
  属于 ABI 兼容验证构建，不是已安装厂商 wheel 的逐 bit 重建。
- 本仓库不分发实验 wheel，只发布源码、测试、补丁和机器可读结果。
- 确定性路径有性能成本，因此补丁只在用户明确请求确定性时改变路由。

详细构建来源和哈希见
[`results/patched-build-receipt.json`](results/patched-build-receipt.json)。

## 文件说明

- [`reproduce_embedding_backward.py`](reproduce_embedding_backward.py)：跨新进程最小复现；
- [`proposed_test_embedding_determinism.py`](proposed_test_embedding_determinism.py)：建议回归测试；
- [`benchmark_embedding_backward.py`](benchmark_embedding_backward.py)：设备同步算子微基准；
- [`patches/embedding-determinism.patch`](patches/embedding-determinism.patch)：最小内核补丁；
- [`results/`](results/)：补丁前后及性能结果。

## 建议提交位置

优先向 [`MooreThreads/torch_musa`](https://github.com/MooreThreads/torch_musa) 提交 PR，并同时附上
复现器、回归测试、最小内核修改、准确环境信息和性能结果。

当前上游提交：
[`MooreThreads/torch_musa#152`](https://github.com/MooreThreads/torch_musa/pull/152)（已进入维护者审阅）。
该 PR 已把路由 guard 与回归测试适配到公开 `main`；current-main 的 MUSA 设备验证明确留给上游 CI
和维护者审阅，不把 v1.3 环境的实测结果冒充为当前主线设备验证。

## 参与贡献与许可证

欢迎提交其他 MUSA 设备上的复现结果、问题报告和聚焦的回归测试改进。提交前请阅读
[`CONTRIBUTING.md`](CONTRIBUTING.md)。

WKK 在本仓库中的原创工作使用 [BSD 3-Clause License](LICENSE) 开源。补丁里为表达修改而保留的
少量上游源码上下文继续遵循其原有版权和许可证；详见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 与 [`AUTHORS.md`](AUTHORS.md)。

## 参考资料

- [torch_musa embedding 实现](https://github.com/MooreThreads/torch_musa/blob/main/torch_musa/csrc/aten/ops/musa/Embedding.mu)
- [PyTorch CUDA embedding backward](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu)
- [PyTorch deterministic algorithms API](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)
