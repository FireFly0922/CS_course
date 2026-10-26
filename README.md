# math2metal

> 从 PyTorch 训练基础到 LLM 推理加速。

## 目标

通过 CS224N、CS336 和 [Datawhale Diy-LLM](https://datawhalechina.github.io/diy-llm/) 掌握：

- 大模型核心结构与训练原理；
- PyTorch、测试、调试与工程组织；
- FLOPs、显存、带宽、benchmark 与 profiler；
- Prefill、Decode、KV Cache 和推理性能指标；
- Triton、memory-efficient attention、编译优化与 serving；
- 一个可用于推理加速 / AI Infra 实习申请的项目。

## 学习主线

```text
PyTorch Training Bridge
        ↓
Attention / Tensor Layout / Tiny Transformer
        ↓
CS336 Basics + Diy-LLM：训练 TinyLM
        ↓
Resource Accounting / Benchmark / Profiler
        ↓
Prefill / Decode / KV Cache
        ↓
Triton / Memory-Efficient Attention / torch.compile
        ↓
MiniServe 推理加速项目
```

## 阶段项目

| 阶段 | 项目 | 核心产物 | 状态 |
|---|---|---|---|
| Phase 0 | PyTorch Training Bridge | 可复现训练程序、测试、shape/stride 与显存笔记 | 🔓 |
| Phase 1 | Transformer Foundations | Attention、Tensor Layout、Tiny Transformer | ⬜ |
| Phase 2 | TinyLM / Diy-LLM Basics | Tokenizer、Transformer、训练循环、checkpoint、generation | ⬜ |
| Phase 3 | Performance Foundations | FLOPs/显存估算、benchmark、profiler 报告 | ⬜ |
| Phase 4 | Inference Core | Naive Generation 与 KV Cache 对照实验 | ⬜ |
| Phase 5 | Systems Performance | Triton、分块 Attention、`torch.compile` | ⬜ |
| Phase 6 | MiniServe | 求职级推理 runtime 与性能报告 | ⬜ |

## 当前进度

- 已完成：CS224N Assignment 2 的 word2vec、Adam、Dropout 基础。
- 当前任务：Phase 0 — PyTorch Training Bridge。
- 下一门槛：能独立写出训练与评估循环，并解释 tensor shape、stride、dtype、device 和梯度。

## 仓库结构

```text
.
├── README.md
├── AI_Labs.md                  # 详细执行计划与验收标准
├── cs224n-spring2024/          # CS224N 课程资料
├── CS224N_projects/            # CS224N 学习产物
├── cs336-2026/                 # CS336 课程资料与作业
└── CS336_projects/             # 主线项目与 MiniServe
```

## 统一交付标准

每个阶段项目包含：

```text
<project>/
├── src/                        # 模块化代码
├── tests/                      # 正确性与边界测试
├── benchmarks/                 # 可复现性能测试
├── configs/                    # 实验配置
├── results/                    # 原始数据与图表
└── README.md                   # 设计、结果、分析与局限
```

每次验收检查：

- 程序可以复现；
- 核心 tensor 的 shape、含义和生命周期清晰；
- 时间复杂度、空间复杂度和主要内存访问清晰；
- 具备自动测试或基线对照；
- 性能实验记录硬件、软件版本、dtype、shape、warmup 和重复次数；
- 结论由测试与 profiler 数据支持。

## MiniServe 最终产物

MiniServe 将包含：

- Transformer 推理数据流；
- Prefill / Decode 分离；
- KV Cache；
- 多请求 batching；
- Triton 或编译优化；
- TTFT、TPOT、tokens/s 和峰值显存测试；
- baseline、优化版本和正确性对照；
- profiler 证据、架构说明和优化日志。

本地环境用于 CPU 正确性测试和小规模开发，服务器用于 CUDA、Triton、Profiler、多 GPU 与最终性能数据。

## 学习资料

- [CS224N](https://web.stanford.edu/class/cs224n/)
- [CS336](https://cs336.stanford.edu/)
- [Datawhale Diy-LLM](https://datawhalechina.github.io/diy-llm/)
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [Triton](https://triton-lang.org/)
