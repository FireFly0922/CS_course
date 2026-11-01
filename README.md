# math2metal

> 从 PyTorch 与 Transformer 基础，走向 LLM 推理框架、rollout runtime 和 Agentic RL。

## 核心目标

这不是一条以手写 kernel 为终点的路线。仓库围绕以下能力组织：

1. 快速掌握大模型架构、训练原理和最小训练闭环；
2. 熟悉 PyTorch，持续提高 Python / PyTorch 工程能力；
3. 重点掌握 LLM 推理框架的请求、调度、缓存、服务和观测；
4. 完成可用于推理加速 / AI Infra 实习申请的系统项目；
5. 将推理引擎能力延伸到 RL rollout、工具调用和多轮 Agent 训练；
6. 在具备 rollout 基础设施视角后学习 CS336 Alignment、RLVR 和 Agentic RL。

## 学习主线

```text
PyTorch 基础
→ Diy-LLM + CS336：Transformer 与最小训练闭环
→ Diy-LLM 第 10 章 + CS336 Inference：推理基础
→ zero-to-sglang：从推理概念到 Mini SGL
→ mini-sglang：阅读并实验真实轻量推理框架
→ 面向 Agentic RL 的 rollout runtime
→ CS336 Alignment + Diy-LLM RLVR
→ Agentic RL
```

Kernel、CUDA 和 Triton 不再作为独立主线，而是贯穿其中的“框架性能素养”：理解 tensor layout、显存/带宽/算术强度、attention backend、benchmark/profile、数值验证，以及 CUDA Graph、融合和量化对框架的约束。

## 阶段路线

| 阶段 | 学习 / 项目 | 核心产物 | 状态 |
|---|---|---|---|
| Phase 0 | PyTorch Training Bridge | 训练/评估循环、autograd、shape/stride、基础测试 | ✅ 基本完成 |
| Phase 1 | Transformer Foundations | MHA、RoPE、tensor layout、Tiny Transformer | 🔓 进行中 |
| Phase 2 | TinyLM Training Loop | tokenizer、Transformer、训练、checkpoint、generation | ⬜ |
| Phase 3 | Inference Foundations | Naive/KV Cache、Prefill/Decode、指标与资源核算 | ⬜ |
| Phase 4 | Build a Mini SGL | 请求生命周期、HTTP、continuous batching、scheduler、Paged/Radix Cache | ⬜ |
| Phase 5 | Read Mini-SGLang | 真实框架源码地图、chunked prefill、overlap、CUDA Graph、TP、backend | ⬜ |
| Phase 6 | Inference Runtime Portfolio | 服务、调度、缓存、观测、SLO 与可复现性能报告 | ⬜ |
| Phase 7 | Rollout Runtime | 异步 rollout、多轮/工具调用、轨迹元数据、backpressure | ⬜ |
| Phase 8 | Alignment → Agentic RL | CS336 Alignment、Diy-LLM RLVR、Agentic RL 实验 | ⬜ |

详细任务、门槛和验收标准见 [AI_Labs/AI_Labs.md](AI_Labs/AI_Labs.md)。项目交付清单见 [CS336_projects/README.md](CS336_projects/README.md)。

## 当前进度与下一节

- Phase 0 已基本完成，不从头重复。
- Phase 1 已学习 Embedding、位置编码概览和单头 Scaled Dot-Product Attention。
- 下一节：**Multi-Head Attention → RoPE → tensor layout**。
- 随后的 Phase 1 任务：RMSNorm、SwiGLU、残差、causal LM 数据流和可过拟合小数据的 Tiny Transformer。

每一节默认采用：

```text
中文理论概念
→ 数学 / shape / 数据流
→ PyTorch API 解释
→ 可运行 toy example
→ 独立练习
→ 测试与后续 review
```

每个重要 PyTorch API 都必须紧跟一个小型使用示例；本地先用 CPU 和小 shape 建立正确性，服务器再做 CUDA、多 GPU 和性能实验。

## 推理框架能力清单

主线项目需要逐步覆盖：

- 请求生命周期、streaming、取消、超时和 backpressure；
- Scheduler、continuous batching、admission control；
- Prefill / Decode、KV Cache 生命周期与容量核算；
- Paged KV Cache、Prefix / Radix Cache；
- Chunked Prefill、overlap scheduling；
- CUDA Graph 的框架使用与 shape / 内存约束；
- attention backend 接口、选择和回退；
- tensor parallel 与多进程请求流；
- OpenAI-compatible API 与可观测性；
- latency、throughput、TTFT、TPOT/ITL、goodput、尾延迟与 SLO。

## 性能素养：必学但不喧宾夺主

必须会：

- 从 kernel 输入输出和 tensor layout 理解框架调用边界；
- 区分 compute-bound 与 memory-bound，估算显存、带宽和算术强度；
- 解释不同 attention backend 的适用条件；
- 正确执行 warmup、同步、重复测量、benchmark 和 profile；
- 验证优化前后数值一致，并保留回归测试；
- 说明 CUDA Graph、融合算子和量化算子对调度与缓存的约束。

不再强制：

- 从零实现高性能 Triton FlashAttention；
- 深入 PTX、warp 或长期手调 block size；
- 完成 CS336 Assignment 2 的全部 kernel TODO；
- 用单个算子速度作为最终项目的核心成果。

## Rollout runtime 为什么位于 RL 之前

Agentic RL 不只是算法问题。训练系统必须反复调用推理引擎生成轨迹，并处理多轮状态、工具调用、超时、失败、模型版本和 reward 元数据。Phase 7 将把 Phase 4–6 的推理能力改造成训练基础设施：

```text
Trainer / Policy Update
        ↓ model version
Rollout Orchestrator
        ↓ batched async requests
Inference Runtime ↔ Tool / Environment
        ↓ trajectories + token/logprob/reward metadata
Verifier / Reward / Replay or Training Data
```

进入 Agentic RL 前，至少应能解释一次请求如何被调度、如何占用/释放 KV Cache、如何度量长尾延迟，以及多轮/工具调用为何会改变 batching 和缓存策略。

## 仓库结构

```text
.
├── AGENTS.md                   # 全仓库教学与协作总规约
├── README.md                   # 总路线与当前入口
├── AI_Labs/
│   ├── AI_Labs.md             # 详细执行计划与阶段验收
│   ├── P0/                    # PyTorch bridge
│   └── P1/                    # Transformer foundations
├── cs224n-spring2024/          # CS224N 课程资料
├── CS224N_projects/            # CS224N 学习产物
├── cs336-2026/                 # CS336 课程资料与正式作业
└── CS336_projects/             # 主线项目与求职级交付物
```

## 通用交付标准

阶段项目按需要包含：

```text
<project>/
├── src/                        # 模块化代码
├── tests/                      # 正确性、边界与回归测试
├── benchmarks/                 # 可复现性能测试
├── configs/                    # 实验配置
├── results/                    # 原始数据与汇总
├── docs/                       # 架构、请求流或实验记录
└── README.md                   # 设计、运行、结果、局限
```

验收统一检查：

- 可在记录的环境和命令下复现；
- 核心 tensor shape、请求状态和资源生命周期清晰；
- 有自动测试、参考实现或 baseline 对照；
- 性能实验记录硬件、版本、模型、dtype、输入/输出长度、并发、warmup 和重复次数；
- 区分 cold start 与 steady state、平均值与 percentile；
- 结论由测试、profiler、trace 或原始数据支持；
- README 能说明设计取舍、失败实验和已知局限。

## 学术诚信

CS336 正式作业的核心实现、数学题和 writeup 由学习者独立完成。AI 只在各作业子目录 `AGENTS.md` 允许的范围内进行翻译、概念讲解、文档定位、测试思路、错误解释和 code review。子目录规约比本文件更严格时，以子目录规约为准。

## 主要资料

- [Stanford CS336 Spring 2026](https://cs336.stanford.edu/)
- [Datawhale Diy-LLM](https://datawhalechina.github.io/diy-llm/)
- [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang)
- [mini-sglang](https://github.com/sgl-project/mini-sglang)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

资料结构最后核对：2026-10-31。zero-to-sglang 仍在持续编写，开始对应阶段前应重新检查章节状态。
