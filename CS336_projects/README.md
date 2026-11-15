# LLM Inference / AI Infra Projects

这个目录存放学习主线的可运行产物。项目组合从 Transformer 与训练闭环出发，重点进入推理框架，并最终连接 Agentic RL rollout runtime。

## 执行顺序

```text
transformer-foundations
→ tinylm
→ inference-foundations
→ cpp-inference-runtime（工程从 Phase 1–2 桥接，Phase 3 执行，Phase 4 服务）
  + mini-sgl-labs（设计对照与 Phase 5 真实框架实验）
→ inference-runtime
→ rollout-runtime
→ agentic-rl-lab
```

| 目录 | 主要能力 | 核心证据 |
|---|---|---|
| `transformer-foundations/` | MHA、RoPE、tensor layout、Transformer block | shape/stride 记录、reference/gradient 测试、Tiny Transformer |
| `tinylm/` | tokenizer、训练闭环、checkpoint、generation | 小数据过拟合、resume、生成样例、测试 |
| `inference-foundations/` | Prefill/Decode、KV Cache、推理指标 | Naive vs cache 正确性/性能、容量核算 |
| `cpp-inference-runtime/` | C++ 桥接、模型执行、缓存、调度、服务；Phase 1–4 递进 | 自写练习、CMake/CTest、数值对照、状态与资源测试、真实基线 |
| `mini-sgl-labs/` | zero-to-sglang 设计对照、Phase 5 mini-sglang 源码与实验 | 请求/资源地图、差异说明、trace 与受控实验 |
| `inference-runtime/` | Phase 6 Inference Runtime Portfolio；优先内存受限本地推理，或 SLO-aware serving | 复用实现，固定 workload、质量检查、内存/延迟、profiler/trace |
| `rollout-runtime/` | 异步 rollout、多轮/工具调用、轨迹与版本管理 | rollout goodput、故障实验、trajectory schema |
| `agentic-rl-lab/` | RLVR 到 Agentic RL | 质量/系统联合报告、可复现实验 |

`phase0-pytorch-bridge/` 若已存在则作为 Phase 0 历史产物保留，不再阻塞当前 Phase 1。

## C++ 入场与交付门槛

本计划更新于 2026-11-13。C++ 熟练度与 WSL2/Linux 工具链尚未验收；以下目录为计划路径，按实际学习创建。

| 范围 | 入场条件 | 个人交付物 |
|---|---|---|
| Foundation 准备 | 当前可开始，缺失 C++ 基础按需补 | 编译/构建、所有权、shape 与失败路径练习 |
| Ch00–03 数据底座 | C0：C++17、CMake/CTest、RAII/owner/view、独立练习验收 | 真实参考基线、加载链说明、Tensor view 与 CPU 算子测试 |
| Ch04–06 模型执行 | C0 + Gate 1；默认 Phase 2 后系统推进 | 小 shape 数值对照、Decoder/生成、位置与 KV 一致性；通过 C1 |
| Phase 4 — C++ Inference Runtime + Mini SGL 对照 | Gate 2、Gate 3 + C1 | 状态、容量、隔离、取消/超时、缓存释放、streaming、真实测量 |
| Phase 5 — Read Mini-SGLang | Gate 4 | 真实框架源码地图与 cache/scheduler/backend 实验 |
| Phase 6 — Inference Runtime Portfolio | Gate 5，选择一个系统问题 | baseline/改动/ablation、测试与完整报告；通过 Gate 6 |
| Phase 7 — Rollout Runtime | Gate 4–6 | 异步轨迹、模型版本、工具状态、故障恢复与有效吞吐 |
| Phase 8 — Alignment → Agentic RL | Gate 7 | RLVR/Agentic RL 实验与质量/系统联合分析 |

详细检查表以 [执行计划第 3.1 节](../AI_Labs/AI_Labs.md#31-cpp-llm-engine-前置条件与分段准入) 为准。参考实现测试通过不能替代自己的 starter 验收。Phase 0 基本完成，Phase 1 进行中，其余阶段未开始；不因纳入新资料而改写实际进度。

`cpp-inference-runtime/` 复用同一工程；`inference-foundations/` 组织跨实现实验；`mini-sgl-labs/` 不重写完整同功能 Python 引擎；`inference-runtime/` 指向所复用实现与 commit，组织求职报告。保留上游许可证和来源，明确个人贡献，正式作业实现不用于参考移植。

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

`inference-runtime/` 默认候选为 **Memory-Bounded Local Inference（内存受限的本地推理）**：复用 C++ runtime，在一个固定设备/模型上研究加载、工作区或 KV 容量管理中的一个问题，记录质量、TTFT、稳态 decode、峰值内存和容量不足后的恢复。端侧设备未确定前只报告实际 CPU/桌面平台结果；移动端/NPU、功耗和温度需要真实设备验证，不能由教学 CPU 结果推断。

备选为 **SLO-aware Serving Lab**：复用已有 C++ runtime 或 mini-sglang 实验层，评估一个框架改进，例如：

- SLO-aware admission / scheduling；
- cache-aware scheduling；
- 长短请求隔离；
- prefix cache 容量与淘汰策略；
- chunked prefill token budget；
- 公平性与 goodput 权衡；
- 取消/超时/backpressure 与资源回收；
- 可观测性和性能回归工具。

只选一个求职主项目，不额外叠加完整端侧和服务项目。不要为了“代码量”重复实现成熟框架已有的全部模型和 kernel；把精力用于问题定义、正确性、实验设计和系统取舍。

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

C++ 项目使用 `CMakeLists.txt`、`include/`、`src/`、`tests/` 与 CTest；无需为了套模板创建 `pyproject.toml`。个人 starter 与上游参考工程分开保存，README 记录二者运行命令和来源。

## 通用项目规则

1. 本地先做 CPU、小 shape、状态机和正确性测试；服务器/WSL2 Linux 做 CUDA、多 GPU 和真实性能实验。
2. 优化前保留 baseline；每次改动都要有可反驳的系统假设。
3. 测试覆盖正常路径、边界、取消、超时、OOM/容量不足、资源释放和数值一致性。
4. 性能结果记录 commit、硬件、驱动/CUDA/PyTorch/框架版本、模型、dtype、TP、backend、请求长度分布、并发、warmup 和重复次数。
5. 分开 cold start / steady state、client / server、平均值 / percentile；至少报告 TTFT、TPOT/ITL、throughput、goodput 和 P95/P99。
6. 保存原始 CSV/JSON/trace 与分析脚本，不只粘贴最终表格。
7. README 记录运行方式、架构、结果、失败实验、适用区间和已知局限。
8. 简历数字只能来自可复现实验，不使用未经验证的估算。

C++/CPU 实验额外记录编译器、构建模式/优化 flags、CMake、线程数及进程 RSS/工作区/KV 内存；CUDA 不适用时标注 N/A。指标按场景选择，固定计时范围，不将 CPU 成本模型模拟值当成真实性能。

## 学术诚信

`cs336-2026/assignments/` 中的正式 CS336 作业与本目录项目严格分离。正式作业的核心实现、数学题和 writeup 由学习者独立完成；AI 必须遵守对应子目录的 `AGENTS.md`，不得把本目录代码改造成作业答案或直接复制进作业目录。

## 资料

- [Stanford CS336 Spring 2026](https://cs336.stanford.edu/)
- [Diy-LLM](https://datawhalechina.github.io/diy-llm/)
- [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang)
- [mini-sglang](https://github.com/sgl-project/mini-sglang)
- [cpp-llm-engine 使用说明](https://github.com/datawhalechina/cpp-llm-engine/blob/main/USAGE.md)
- [llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode)

cpp-llm-engine 与 llm-algo-leetcode 核对：2026-11-13（文档及部分源码，未运行完整测试）；其他课程结构沿用 2026-10-31 记录，阶段启动前复核。
