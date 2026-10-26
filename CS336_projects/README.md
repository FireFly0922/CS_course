# CS336 / Inference Acceleration Projects

这个目录存放学习主线的可运行产物，并逐步组合成用于推理加速 / AI Infra 实习申请的项目组合。

## 执行顺序

```text
phase0-pytorch-bridge
    ↓
tinylm
    ↓
inference-core
    ↓
systems-performance
    ↓
miniserve
```

| 目录 | 主要能力 | 核心证据 |
|---|---|---|
| `phase0-pytorch-bridge/` | PyTorch、autograd、训练循环、shape / stride | loss 曲线、测试、参数/显存核算 |
| `tinylm/` | tokenizer、Transformer、AdamW、checkpoint、generation | CS336 / Diy-LLM 选定测试与生成样例 |
| `inference-core/` | Prefill、Decode、KV Cache、推理指标 | Naive vs KV Cache 正确性与性能对照 |
| `systems-performance/` | profiler、Triton、tiled attention、`torch.compile` | trace、benchmark 数据和优化日志 |
| `miniserve/` | 推理 runtime、batching、cache 管理和性能工程 | 求职级 README、架构、测试、结果与局限 |

## 通用项目规则

1. 先在本地用 CPU 和小 shape 验证正确性，再在服务器生成 CUDA 性能数据。
2. 优化前必须保留 baseline；每次优化都要有可反驳的性能假设。
3. 性能结果必须记录硬件、软件版本、shape、dtype、warmup 和重复次数。
4. 必须分开首次运行开销与稳定态，分开 latency、throughput、显存和正确性。
5. 每个子项目都要包含可重复命令、自动测试、结果文件和已知局限。

## 学术诚信

CS336 正式作业的核心实现保留给学习者独立完成。AI 负责翻译、概念讲解、API 查询、测试设计、错误定位和 code review。

