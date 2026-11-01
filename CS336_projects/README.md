# LLM Inference / AI Infra Projects

这个目录存放学习主线的可运行产物。项目组合从 Transformer 与训练闭环出发，重点进入推理框架，并最终连接 Agentic RL rollout runtime。

## 执行顺序

```text
transformer-foundations
→ tinylm
→ inference-foundations
→ mini-sgl-labs
→ inference-runtime
→ rollout-runtime
→ agentic-rl-lab
```

| 目录 | 主要能力 | 核心证据 |
|---|---|---|
| `transformer-foundations/` | MHA、RoPE、tensor layout、Transformer block | shape/stride 记录、reference/gradient 测试、Tiny Transformer |
| `tinylm/` | tokenizer、训练闭环、checkpoint、generation | 小数据过拟合、resume、生成样例、测试 |
| `inference-foundations/` | Prefill/Decode、KV Cache、推理指标 | Naive vs cache 正确性/性能、容量核算 |
| `mini-sgl-labs/` | 请求生命周期、HTTP、scheduler、continuous batching、Paged/Radix Cache | 状态机测试、请求 trace、并发 benchmark |
| `inference-runtime/` | 真实框架实验、SLO-aware serving、可观测性 | 架构文档、workload、P50/P95/P99、profiler/trace |
| `rollout-runtime/` | 异步 rollout、多轮/工具调用、轨迹与版本管理 | rollout goodput、故障实验、trajectory schema |
| `agentic-rl-lab/` | RLVR 到 Agentic RL | 质量/系统联合报告、可复现实验 |

`phase0-pytorch-bridge/` 若已存在则作为 Phase 0 历史产物保留，不再阻塞当前 Phase 1。

## 当前入口

当前阶段为 `transformer-foundations/`：

```text
Multi-Head Attention
→ RoPE
→ tensor layout / stride / contiguous
→ RMSNorm + SwiGLU + residual
→ Tiny Transformer
```

教学顺序必须是“中文理论 → API 解释 → 可运行 toy example → 学习者独立练习 → tests/review”。每个重要 PyTorch API 后应紧跟小型示例。

## 推理项目的范围

推理项目优先解决框架级问题：

- request lifecycle、streaming、cancel、timeout；
- scheduler、admission、continuous batching；
- Prefill/Decode 与 KV Cache 生命周期；
- Paged KV Cache、Prefix/Radix Cache；
- chunked prefill、overlap scheduling；
- CUDA Graph 的框架集成和限制；
- attention backend 选择与回退；
- tensor parallel 和多进程流；
- API、metrics、logs、traces、SLO 与 backpressure。

Kernel / Triton 只承担性能素养训练：理解接口、layout、带宽/算术强度、profile 和数值验证。项目不要求从零实现高性能 FlashAttention，也不以单一 kernel 加速比作为主要成果。

## 推荐的求职主项目

`inference-runtime/` 默认做成 **SLO-aware Serving Lab**：在自建 Mini SGL 或 mini-sglang 实验层上实现并评估一个框架改进，例如：

- SLO-aware admission / scheduling；
- cache-aware scheduling；
- 长短请求隔离；
- prefix cache 容量与淘汰策略；
- chunked prefill token budget；
- 公平性与 goodput 权衡；
- 取消/超时/backpressure 与资源回收；
- 可观测性和性能回归工具。

不要为了“代码量”重复实现成熟框架已有的全部模型和 kernel；把精力用于问题定义、正确性、实验设计和系统取舍。

## Rollout Runtime 交付范围

`rollout-runtime/` 把 serving runtime 扩展为训练基础设施：

- batched async rollout / generation group；
- 多轮会话与 tool call 状态机；
- token、logprob、mask、reward、tool event、termination reason；
- cancellation、timeout、retry、幂等和部分失败；
- policy/model version 与 in-flight rollout 一致性；
- trainer / inference / environment / verifier 接口；
- rollout goodput、长尾、backpressure 和资源隔离。

## 通用项目结构

```text
<project>/
├── README.md
├── pyproject.toml
├── src/
├── tests/
├── benchmarks/
├── configs/
├── results/
├── profiles/
└── docs/
```

目录按项目需要创建，不为了形式保留空目录。

## 通用项目规则

1. 本地先做 CPU、小 shape、状态机和正确性测试；服务器/WSL2 Linux 做 CUDA、多 GPU 和真实性能实验。
2. 优化前保留 baseline；每次改动都要有可反驳的系统假设。
3. 测试覆盖正常路径、边界、取消、超时、OOM/容量不足、资源释放和数值一致性。
4. 性能结果记录 commit、硬件、驱动/CUDA/PyTorch/框架版本、模型、dtype、TP、backend、请求长度分布、并发、warmup 和重复次数。
5. 分开 cold start / steady state、client / server、平均值 / percentile；至少报告 TTFT、TPOT/ITL、throughput、goodput 和 P95/P99。
6. 保存原始 CSV/JSON/trace 与分析脚本，不只粘贴最终表格。
7. README 记录运行方式、架构、结果、失败实验、适用区间和已知局限。
8. 简历数字只能来自可复现实验，不使用未经验证的估算。

## 学术诚信

`cs336-2026/assignments/` 中的正式 CS336 作业与本目录项目严格分离。正式作业的核心实现、数学题和 writeup 由学习者独立完成；AI 必须遵守对应子目录的 `AGENTS.md`，不得把本目录代码改造成作业答案或直接复制进作业目录。

## 资料

- [Stanford CS336 Spring 2026](https://cs336.stanford.edu/)
- [Diy-LLM](https://datawhalechina.github.io/diy-llm/)
- [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang)
- [mini-sglang](https://github.com/sgl-project/mini-sglang)

资料结构最后核对：2026-10-31。
