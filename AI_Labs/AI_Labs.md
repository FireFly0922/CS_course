# LLM 推理框架与 Agentic RL 执行计划

> 版本：2026-11-13。Phase 0 已基本完成；当前从 Phase 1 的 Multi-Head Attention、RoPE 和 tensor layout 继续。纳入 cpp-llm-engine 工程实践，不改变学习者先实现、AI review 的教学协议。

## 1. 最终目标

建立一条从模型原理到推理系统、再到 Agentic RL 基础设施的完整能力链：

```text
PyTorch / Tensor
→ Transformer / 最小训练闭环
  （并行 C++ 工程与 Tensor 所有权桥接）
→ 自回归推理 / KV Cache / C++ CPU 执行
→ C++ Request / Scheduler / Cache / Serving + Mini SGL 设计对照
→ 真实推理框架源码与实验
→ Rollout Runtime
→ RLVR / Agentic RL
```

最终求职证据不是“我写过一个快 kernel”，而是：

- 能解释并实现请求从 API 到 token stream 的完整生命周期；
- 能设计 scheduler、continuous batching 和 KV Cache 管理；
- 能用 TTFT、TPOT/ITL、throughput、goodput、尾延迟和 SLO 评估系统；
- 能阅读并修改 mini-sglang 这类真实推理框架；
- 能把推理 runtime 改造成支持异步、多轮、工具调用的 RL rollout 基础设施；
- 能用测试、benchmark、profiler、trace 和原始数据支撑结论。

## 2. 教学与实践协议

每个重要知识点按以下顺序完成：

1. 中文直觉解释，并标注英文术语；
2. 数学、tensor shape、状态或请求数据流；
3. 解释相关 PyTorch / Python API 的参数、返回值和常见错误；
4. 每个重要 API 紧跟一个 CPU、小 shape、可运行 toy example；
5. 学习者完成独立练习或首版实现；
6. 用单元测试、边界测试、日志或 trace 定位问题；
7. AI 做 code review，检查理解、正确性和工程质量；
8. 说明它与后续推理框架或 rollout runtime 的连接。

AI 可负责中文翻译、概念讲解、API 解释、toy example、测试设计、错误解释、普通项目协作和 code review。CS336 正式作业必须遵守各作业目录更严格的 `AGENTS.md`：核心实现、数学题和 writeup 由学习者独立完成，AI 不写作业代码或伪代码、不修改作业目录。

## 3. 课程与项目映射

| 阶段 | 主要资料 | 使用方式 | 核心目标 |
|---|---|---|---|
| Phase 0 | Diy-LLM Ch3 | 已完成部分作为查漏资料 | PyTorch 训练原语 |
| Phase 1 | Diy-LLM Ch4、CS336 Basics、算法库组件练习；C++ Foundation | 模型基础为主，少量 C++ 桥接 | Transformer forward、C0 工程准备 |
| Phase 2 | Diy-LLM Ch2–4、CS336 Assignment 1；C++ Ch00–03 | 自建 TinyLM + C++ 基线/数据底座；正式作业独立完成 | 最小训练闭环、模型加载与 Tensor |
| Phase 3 | Diy-LLM Ch10、CS336 Lecture 10；C++ Ch04–06 | PyTorch/C++ 小规模对照；算法库按需补充 | Prefill/Decode/KV Cache/指标、C1 |
| Phase 4 | cpp-llm-engine Ch07–10、zero-to-sglang Part I–II | 复用 C++ 工程，Mini SGL 用于设计对照和差异实验 | 请求、调度、缓存、服务 |
| Phase 5 | mini-sglang、zero-to-sglang Part III | 源码地图 + 可控实验 | 真实轻量框架 |
| Phase 6 | 自建 portfolio project | 围绕一个系统问题做扩展 | AI Infra 求职项目 |
| Phase 7 | 自建 rollout runtime | 推理服务连接环境/工具/trainer | RL 基础设施 |
| Phase 8 | CS336 Assignment 5、Diy-LLM Ch13–14 | 正式作业独立完成 + 自建实验 | RLVR → Agentic RL |

说明：Diy-LLM 章节映射沿用 2026-10-31 核对记录；进入对应阶段前复核。cpp-llm-engine 的章节与练习入口于 2026-11-13 核对；只使用当前阶段所需部分。llm-algo-leetcode 简称“算法库”，不是另一条必须完整打卡的课程。

## 3.1 cpp-llm-engine 前置条件与分段准入

这是个人能力验收，不是活动官方 Task 或报名要求。已知 Phase 0 基本完成；C++ 水平、WSL2/Linux、编译器和设备资源均尚未验收，不能由会写 Python 推定已具备。

### 现在可以开始：准备，不必等待整个 Transformer 课程结束

- 先继续 MHA → RoPE → layout，保持当前教学入口。
- 并行检查基础 C++：函数、循环、引用/指针、`std::vector`、类、构造/析构、编译错误；缺哪项用小例子补哪项。
- 工具链采用 WSL2 Ubuntu 或远程 Linux：C++17 编译器、CMake 3.20+、CTest、Git。GPU/CUDA 不是 CPU 准入条件。
- 尚未选定模型前不承诺固定 RAM 下限。实际基线前检查磁盘、可用内存、权重大小、运行时开销与预期 KV；记录模型文件及版本。
- 不需要提前学完 CUDA、Triton、高级模板、多 GPU、RL 或整套 C++ 教材。

### C0：进入 Chapter00–03 工程实践的门槛

- [ ] 能在 Linux/WSL2 中独立构建并运行 C++17 程序，查看编译/运行错误。
- [ ] 能用 CMake 创建 library、可执行程序和测试 target，并运行 CTest；能定位一条故意失败的断言。
- [ ] 能解释对象生命周期、RAII（资源获取即初始化）、移动与拷贝、owner 与 view 的区别；不会让 view 指向已经释放的存储。
- [ ] 独立完成或用已有代码证明 FQ01（所有权）、FQ02（shape）、FQ04（构建测试）、FQ05（错误路径）对应能力。
- [ ] 保存自己的实现、测试命令和错误定位记录；仅运行参考题库不计通过。

C0 通过即可做 Ch00 基线与 Ch01–03 数据底座，不要求 Gate 1/2。FQ03/FQ08（view/Tensor）、FQ06（MatMul）、FQ12（workspace）和 FQ17（工程整合）随项目按需推进，不把所有 Foundation 练习堆成入门门槛。入口见 [Foundation 练习说明](https://github.com/datawhalechina/cpp-llm-engine/blob/main/exercises/foundation/README.md)。

### 模型数学门槛：进入 Chapter04–06

至少通过 C0 + Gate 1：能从 token id 追到 logits，理解 MHA/GQA 的头关系、RoPE 位置、causal mask、RMSNorm、SwiGLU 和残差。GQA 在 MHA 之后用小例子补齐。

本计划默认先完成 Phase 2 的最小训练闭环，再在 Phase 3 系统推进 Decoder/Generation；完成模型组件后可以预读对应 C++ 实现。最小训练闭环是个人能力目标，不是运行预训练模型的技术依赖，更不要求先做完所有 CS336 正式作业。

### C1：进入 Chapter07–10 runtime 的门槛

- [ ] C0、Gate 1、Gate 2、Gate 3 已通过；单请求生成确实正确。
- [ ] 固定模型、tokenizer、prompt、权重/dtype 与 greedy 配置；验证完整前向与缓存 decode 的 logits 容差、位置偏移和输出行为。
- [ ] 能解释 Prefill→Decode、EOS/max tokens、请求终态与资源释放；覆盖单 token、不同前缀长度及异常输入。
- [ ] 能核算每层 K/V 的 shape 与字节数，区分模型权重、工作区和请求状态。
- [ ] 有真实基线的运行记录，能区分加载/预热/稳态，解释 TTFT、TPOT/ITL 和吞吐的计时口径。

量化与不同执行后端允许数值容差，不把所有配置的文本逐字相等当成通用标准；先用小 shape/固定输入核对数值，再检查受控 greedy 序列。达不到 C1 就在当前执行链排错，不用更复杂的 scheduler 掩盖问题。

### 时间与活动边界

当前暂按 2/3 模型基础、1/3 C++ 桥接分配时间，比例随验收调整。一个学习单元约 60–90 分钟只是排期单位，不承诺固定天数掌握。连续两个单元卡在同一前置问题时，缩小到最小例子定位，再继续主线。

组队打卡只能复用本计划的实现、测试与实验记录；参考工程运行、个人独立练习、原创系统改动分别标注。本期 Task0–6 与仓库 CPU 六任务的对应关系、助教 review 和结课要求尚未确认，不以活动两周期限代替阶段验收。

### 路径与去重

- `CS336_projects/cpp-inference-runtime/`：同一工程积累 bridge → execution → serving；不是每阶段重开一个引擎。
- `CS336_projects/transformer-foundations/`、`tinylm/`：保留原 PyTorch 产物，可为小规模 C++ 数值对照提供参考。
- `CS336_projects/inference-foundations/`：只保存跨实现对照、容量核算和实验记录，复用模型代码。
- `CS336_projects/mini-sgl-labs/`：zero-to-sglang/mini-sglang 的源码地图、差异实验与缺失能力补充，不完整重写第二个同功能引擎。
- `CS336_projects/inference-runtime/`：后期求职项目的实验与报告入口，复用 C++ 工程或真实框架，明确实际代码位置。

以上是计划路径，按首次实践创建；不因更新文档而声称已有实现。上游教程代码保留来源、许可证、commit；个人练习与改动可辨识。正式作业不作为参考实现来源。

## 4. 贯穿式性能素养

性能内容不再单独占据主线，而是在遇到真实框架问题时按需学习。

### 必须掌握

- kernel 的输入输出、tensor layout、contiguous/stride 和框架调用边界；
- compute-bound、memory-bound、显存、带宽、FLOPs、Bytes moved 和算术强度；
- Prefill 与 Decode 为什么常有不同瓶颈；
- FlashAttention、FlashInfer、Triton/FA 等 backend 的职责、约束和选择条件；
- warmup、GPU synchronize、重复测量、percentile、profiler 和 trace；
- baseline 与优化版的数值一致性、容差和回归测试；
- CUDA Graph 对静态 shape、内存地址、batch bucket 和调度的限制；
- 融合、量化和自定义算子对模型兼容性与框架回退路径的影响。

### 不再作为门槛

- 从零实现高性能 Triton FlashAttention；
- 深入 PTX、warp 调度或长期手调 block size；
- 完成 CS336 Assignment 2 的所有 kernel TODO；
- 用单一 kernel 加速比充当最终项目成果。

Assignment 2 可选择性学习 profiler、benchmark、distributed 和 backend 概念。若学习者自愿挑战 kernel，必须设置明确时间盒和停止条件，不得阻塞 Phase 3–5。

## 5. Phase 0 — PyTorch Training Bridge

**状态：✅ 基本完成。**

保留成果：tensor、autograd、`nn.Module`、训练/评估循环、`zero_grad`、loss、AdamW、shape/stride、参数与基础内存核算。

不默认重做。只有出现以下阻塞时，才用最小示例回补：

- 不会判断梯度是否存在或是否累积；
- 混淆 logits、probability 和 cross entropy；
- 无法解释 `train()`、`eval()`、`no_grad()`；
- 无法追踪 shape、dtype、device 或 stride。

## 6. Phase 1 — Transformer Foundations

**状态：🔓 进行中。**

### 已完成

- Embedding；
- 位置编码概览；
- 单头 Scaled Dot-Product Attention。

### 下一节：Multi-Head Attention → RoPE → tensor layout

#### 1A. Multi-Head Attention

数据流必须逐步解释：

```text
x [B,T,C]
→ qkv projection [B,T,3C]
→ split q/k/v
→ reshape [B,T,H,D]
→ transpose [B,H,T,D]
→ scores [B,H,T,T]
→ causal softmax
→ values [B,H,T,D]
→ transpose + contiguous + reshape [B,T,C]
→ output projection
```

紧跟的 API toy examples：`nn.Linear`、`chunk`/`split`、`view`/`reshape`、`transpose`/`permute`、`contiguous`、broadcasting、`masked_fill`、`softmax`、`matmul` 或 `einsum`。

独立练习：学习者实现小 shape 的 causal MHA，并对照 PyTorch SDPA 或可信 reference 检查输出和梯度。

#### 1B. RoPE

- 从二维旋转理解相对位置信息；
- 说明 RoPE 应用于 Q/K 而非 V；
- 跟踪 `[B,H,T,D]` 中 position 和 pair 维度；
- 比较完整序列与带 `start_pos` 的增量 decode；
- toy example 验证旋转保持向量范数，并验证位置偏移行为。

#### 1C. Tensor layout

- 用 `shape`、`stride()`、`is_contiguous()` 观察每一步；
- 解释为何 transpose 后直接 `view` 可能失败或语义错误；
- 比较 `[B,H,T,D]` 与推理 backend / KV Cache 可能需要的 layout；
- 只讨论与 attention 和 cache 接口直接相关的内存含义，不深入 kernel 微架构。

### 后续任务

- RMSNorm、SwiGLU/MLP、残差连接；
- `[B,T] → [B,T,C] → [B,T,V]`；
- causal next-token prediction 与 target shift；
- 组合为可过拟合极小数据集的 Tiny Transformer。

### 产物

路径：`CS336_projects/transformer-foundations/`

- MHA + RoPE toy labs；
- tensor layout 实验记录；
- Tiny Transformer；
- forward/gradient/causal mask 测试；
- 中文 README：从 token id 到 logits 的 shape 跟踪。

### Gate 1

进入 Phase 2 前，学习者能独立回答：

- head 拆分与合并的每一步 shape/stride 是什么；
- RoPE 如何处理训练 full sequence 与 decode 单步位置；
- causal mask 在 softmax 前如何广播；
- 为什么 `contiguous()` 不是“为了让代码不报错”这么简单；
- 一个 token 如何从 id 变成 logits。

## 7. Phase 2 — TinyLM：Transformer 与最小训练闭环

### C++ 衔接

通过 C0 后可在本阶段推进 cpp-llm-engine Ch00–03：先建立外部真实推理基线，再追踪模型输入/权重，练习 Tensor view 与 CPU 算子。模型训练任务和 C++ 数据底座独立验收；不在此阶段展开多个模型、MoE、CUDA 或完整服务。保存基线配置、所有权示意、shape/stride/offset 检查和一个与 PyTorch 对照的 CPU 小算子。

### 目标

把 Diy-LLM 与 CS336 Basics 的知识组合成可训练、可保存、可恢复、可生成的小语言模型，并提升 Python/PyTorch 工程能力。

### 项目路径

`CS336_projects/tinylm/`

### 模块

```text
Tokenizer / ids
→ Dataset packing + target shift
→ Transformer LM
→ Cross entropy
→ AdamW + schedule
→ Training / validation
→ Checkpoint + resume
→ Generation + evaluation
```

### 工程练习

- dataclass/config、模块边界、类型标注和错误信息；
- `state_dict`、checkpoint 原子保存和 resume；
- seed、determinism 和最小可复现实验；
- pytest 参数化、小数据过拟合和梯度 sanity checks；
- CLI 与 README 的可重复命令。

### 验收

- tokenizer 能 encode/decode；
- input/target 正确错开一个 token；
- 小数据集能过拟合；
- checkpoint 恢复模型、优化器、scheduler、step 和 RNG 状态；
- greedy generation 可运行；
- 模型配置、参数量、训练曲线、失败案例有记录。

### 学术诚信边界

自建 TinyLM 可以协作；`cs336-2026/assignments/assignment1-basics/` 中正式作业严格遵守其 `AGENTS.md`，不得把自建实现直接转成作业答案。

### Gate 2

能从 batch 构造追踪到 loss/backward/checkpoint/generation，并能在不依赖大型框架的情况下定位一次训练不收敛问题。

## 8. Phase 3 — Inference Foundations

### 资料

- Diy-LLM 第 10 章推理；
- CS336 Spring 2026 Lecture 10 Inference；
- zero-to-sglang Part I 作为后半段衔接。

### 项目路径

`CS336_projects/inference-foundations/`

### 学习任务

1. 自回归 generation：每步完整重算前缀的 baseline。
2. KV Cache：每层 K/V shape、写入位置、增长、释放与字节数。
3. 分离 Prefill 与 Decode，解释 compute/memory 特征。
4. 理解 request latency、TTFT、TPOT/ITL、E2E latency、throughput 和 goodput。
5. 设计离线与在线 benchmark，理解平均值、P50/P95/P99 和 SLO。
6. 认识 PagedAttention、quantization、speculative decoding，但不在此阶段全部实现。

### 实验

- Naive generation vs KV Cache generation；
- prompt length、generation length、batch、dtype 对结果的影响；
- greedy decode 下 token 输出一致；
- cache 理论大小与实测峰值显存对照；
- 将 profiler 结论限制在“解释瓶颈”，不转为无限 kernel 优化。

### Gate 3

能说明为什么 Prefill 与 Decode 需要不同调度/优化，能计算 KV Cache 容量，并能避免把 tokens/s、TTFT 和 TPOT 混为一个指标。

### C++ 执行链与 C1

通过模型数学门槛后，按 cpp-llm-engine Ch04–06 衔接 Decoder、量化 Linear 与生成。只选一个稠密模型；先理解小 shape 的浮点数学与参考路径，再理解量化表示和误差。学习量化服务于模型加载/内存/带宽，不另开全量量化格式研究。

在 `cpp-inference-runtime/` 保存执行链与测试，在 `inference-foundations/` 保存对照报告。Gate 3 检查推理概念；C1 额外检查实际执行证据，两者都通过才能进入 Phase 4。详见第 3.1 节。

## 9. Phase 4 — C++ Inference Runtime + Mini SGL 对照

### 目标

从推理概念进入框架工程，在 cpp-llm-engine Ch07–10 对应的同一工程中逐步增加缓存、调度、测量和服务。zero-to-sglang Part I–II 用于设计对照；算法库用于有明确问题的机制实验。入场要求 Gate 3 + C1，不按 kernel 类型组织任务。

### 项目路径

实现：`CS336_projects/cpp-inference-runtime/`；设计对照与差异实验：`CS336_projects/mini-sgl-labs/`。

### 顺序与产物

1. **框架地图**：API、tokenizer、scheduler、model runner、detokenizer 的边界。
2. **请求生命周期**：arrival → tokenize → queue → prefill → decode → stream/finish/cancel。
3. **Forward + generation**：连接 Phase 2/3 的模型和 KV Cache。
4. **HTTP 与并发请求**：隔离、streaming、取消、超时、异常。
5. **Continuous batching + Scheduler**：running/waiting 状态、token budget、admission 和抢占/延迟权衡。
6. **Paged KV Cache**：page/block 分配、映射、回收、碎片和 OOM 行为。
7. **Radix/Prefix Cache**：共享前缀、引用/淘汰、命中率与正确性。
8. **多进程与 Tensor Parallel**：先理解请求/控制流，再做服务器实验。
9. **Speculative decoding**：作为可选扩展，关注框架协议与收益条件。

前七项优先在已有 C++ 工程形成完整请求链。Prefix Cache 不必立即写成 Radix Tree，先验证共享、引用、淘汰和失效语义；Radix 的具体实现可以在 Phase 5 阅读和实验。多进程/TP 在这里完成请求流设计说明，真实多 GPU 运行留到 Phase 5。超出 C++ 教程当前能力的内容，通过独立小实验或真实框架补齐，不能仅凭章节标题标为已完成。

### 验收

- 一张带状态和资源所有权的请求生命周期图；
- scheduler 的确定性单元测试；
- page 分配/释放/复用/OOM 测试；
- 并发、取消和 streaming 测试；
- 在多个请求分布下报告 TTFT、TPOT、throughput、goodput 和 P95/P99；
- 能解释调度策略如何改变延迟/吞吐，而不是只给最终数字。

### Gate 4

能从一条 API 请求追踪到每个 token 返回，能解释请求状态、KV page 所有权和调度决策。

## 10. Phase 5 — mini-sglang：真实轻量推理框架

完成 Phase 4 的 C++ runtime 后，以 mini-sglang 进行真实框架源码阅读与实验，不要求先完整重写 Python Mini SGL。阅读范围包括 API、scheduler、engine、backend、KV、CUDA Graph、TP、Radix Cache、Chunked Prefill 和 Overlap Scheduling；具体模块路径与功能在启动本阶段时按所选 commit 复核。

源码地图和实验统一放在 `CS336_projects/mini-sgl-labs/`。每项能力记录它在 C++ 工程与真实框架中的不同约束，避免将 CPU 教学结果外推为 GPU 服务性能。

### 推荐源码阅读顺序

```text
server / message
→ tokenizer + detokenizer
→ scheduler
→ engine + context
→ model / layers
→ kvcache
→ attention backends
→ cuda graph
→ tensor parallel / process communication
→ benchmark and tests
```

不要从 `kernel/` 开始，也不要一开始逐文件通读。

### 四组必做实验

1. **请求流 trace**：为一个请求记录跨进程/模块事件和状态变化。
2. **Cache 实验**：对比 naive/radix，改变共享前缀比例和容量，观察命中、淘汰、延迟。
3. **调度实验**：改变并发、prompt/output 分布和 chunked prefill 配置，观察 TTFT/TPOT/goodput/尾延迟。
4. **执行实验**：比较 CUDA Graph 开关、attention backend 或 TP 配置；记录兼容条件与回退。

### 重点理解

- Chunked Prefill 如何避免长 prompt 阻塞 decode，并如何分配 token budget；
- Overlap Scheduling 如何隐藏 CPU 调度/通信，但引入何种状态一致性要求；
- CUDA Graph capture/replay 对 batch shape、内存地址和模型路径的要求；
- attention backend 是框架接口选择问题，不要求重写其高性能 kernel；
- TP 下 scheduler、engine 和请求结果如何保持一致。

### Gate 5

能给陌生模块定位到其上游/下游，能设计一个不破坏 correctness 的小改动，并用测试与 trace 验证。

## 11. Phase 6 — Inference Runtime Portfolio

### 目标

交付一个聚焦“框架问题”的求职级项目。复用 `cpp-inference-runtime/` 或 mini-sglang fork/实验层，不重新造整个 vLLM/SGLang。`inference-runtime/` 作为统一实验/报告入口，README 指明实际实现和版本。

### 主项目选择：优先端侧约束，保留 serving 路线

路径：`CS336_projects/inference-runtime/`

根据目前端侧兴趣，默认候选为 **Memory-Bounded Local Inference（内存受限的本地推理）**：在固定设备、模型、质量要求下，选择加载/工作区/KV 容量管理中的一个系统问题，比较首次响应、稳态 decode、峰值内存与失败回收。报告不同上下文长度下的容量和延迟取舍；先定位瓶颈再改实现，不要求全面手写 kernel。

若选 CPU/桌面设备，只宣称该平台结果；移动端/NPU 是目标设备确定后的选做扩展，需要额外验证构建、后端、搬运与实际设备行为。功耗/温度只有真实测量后才进入结论。

另一候选为 **SLO-aware Serving Lab**：选择 admission、缓存策略、长短请求隔离等一个明确问题。两个候选只做一个主项目，不额外叠加完整端侧和服务项目。两条路线均保留 Phase 4 的请求/缓存/取消/观测能力，便于进入 rollout。

已有 runtime 的复用范围（按项目问题选择扩展，不要求重复实现）：

- OpenAI-compatible 或清晰的生成接口；
- request lifecycle 与状态机；
- continuous batching scheduler；
- Paged KV + prefix/radix cache；
- chunked prefill；
- 取消、超时、backpressure；
- metrics、结构化日志和简单 trace；
- workload generator 与可复现 benchmark；
- 至少一个明确框架改进：例如 SLO-aware admission、cache-aware scheduling、长短请求隔离或公平性策略。

### 求职验收

1. README 能让陌生读者安装、运行和复现实验。
2. 架构文档说明模块、请求流、状态和资源所有权。
3. 正确性测试覆盖缓存、调度、取消、OOM 和边界条件。
4. workload 同时包含固定 shape 和分布式请求长度/到达率。
5. 报告 TTFT、TPOT/ITL、throughput、goodput、P50/P95/P99、峰值显存。
6. 记录硬件、软件、模型、dtype、并发、输入/输出分布、warmup 和统计方式。
7. 通过 baseline、优化版、ablation 和 profiler/trace 说明收益来源与失效区间。
8. 简历只使用真实、可复现数据。

CPU 项目记录 RSS/峰值工作区/KV 字节等实际内存指标；GPU 项目补显存指标，明确 allocator allocated/reserved 与进程内存的差别。模拟数据只能说明机制。上述验收通过记为 Gate 6。

## 12. Phase 7 — 面向 Agentic RL 的 Rollout Runtime

入场：Gate 4、Gate 5、Gate 6 通过。优先以清晰推理服务接口连接已有 runtime；如需 logprob、模型版本或更新能力，先验证后端支持，不默认 C++ 教学引擎已具备 trainer 所需全部能力。

### 目标

理解“推理引擎如何成为 RL 训练基础设施”，而不只是给用户提供 chat completion。

### 项目路径

`CS336_projects/rollout-runtime/`

### 架构

```text
Trainer / Policy Version Manager
        ↓ rollout spec
Rollout Orchestrator
        ├── Inference Runtime
        ├── Tool / Environment Workers
        └── Verifier / Reward
        ↓
Trajectory Store / Training Batch
```

### 学习任务

- 异步批量 rollout 与 generation group；
- 多轮会话状态和工具调用状态机；
- token、logprob、mask、tool event、reward、termination reason 等轨迹 schema；
- 取消、超时、重试、幂等和部分失败；
- policy/model version 与 in-flight rollout 一致性；
- backpressure、长尾请求、资源隔离与 rollout goodput；
- prefix/KV 复用的收益和版本/上下文失效规则；
- trainer、inference、tool、verifier 之间的接口与可观测性。

### 实验

- 单轮 vs 多轮；无工具 vs 带工具；短尾 vs 长尾环境延迟；
- 不同并发和 generation group size；
- timeout/retry 对吞吐、样本有效率和重复副作用的影响；
- policy version 切换期间的一致性检查；
- 端到端 rollout goodput：单位时间内可用于训练的有效轨迹/token。

### Gate 7

能解释为什么在线 serving 的 tokens/s 不能直接代表 RL rollout 效率，并能追踪一条轨迹从 prompt 到训练样本的全生命周期。

## 13. Phase 8 — CS336 Alignment + Diy-LLM RLVR → Agentic RL

### 学习顺序

1. Diy-LLM 第 13 章：预训练、SFT、DPO/RLHF 流程地图；
2. CS336 Assignment 5 Alignment and Reasoning RL（严格独立完成正式作业）；
3. Diy-LLM 第 14 章：RLVR、GRPO、verifier 和 process reward；
4. 自建小规模 RLVR 实验；
5. 引入多轮环境与工具调用，形成 Agentic RL 实验。

### 系统关注点

- rollout 与训练的同步/异步方式；
- sampling、logprob 和 importance/version 一致性；
- verifier/reward 的批处理与故障；
- trajectory filtering、去重、长度和终止原因；
- 工具副作用、sandbox、超时和可复现性；
- 训练指标与系统指标联合分析。

### 最终产物

- 一个小规模、可复现的 RLVR → Agentic RL 实验；
- rollout runtime 架构和数据契约；
- 算法曲线 + 系统效率 + 失败案例；
- 明确区分模型质量提升、采样变化和系统吞吐变化。

## 14. 本地与服务器分工

### 本地

- CPU、小 shape 和 deterministic correctness；
- C++ 工程在 WSL2/Linux 完成，含 CMake/CTest、资源生命周期与必要的 sanitizer 检查；
- PyTorch API toy examples；
- scheduler/cache 状态机测试；
- mock tool/environment；
- 文档、类型检查和 Git。

### 服务器 / WSL2 Linux

- mini-sglang 安装与真实模型；
- CUDA、attention backend 和 CUDA Graph；
- 多 GPU / tensor parallel；
- profiler、trace 和性能实验；
- 最终 benchmark 与 rollout 压测。

Mini-SGLang 依赖 Linux/CUDA，Windows 原生环境不作为目标平台。

## 15. 性能实验记录模板

每次实验至少记录：

```text
commit / config / command
hardware / driver / CUDA / PyTorch / framework version
CPU threads / compiler / build type / flags / CMake / OS (C++/CPU experiments)
model / dtype / TP size / attention backend
request arrival pattern / concurrency
prompt length distribution / output length distribution
cache and scheduler settings
warmup / duration / repetitions
TTFT / TPOT or ITL / E2E / throughput / goodput
P50 / P95 / P99 / peak memory
correctness checks / errors / timeout / cancellation count
raw result path / analysis script / conclusion / limitation
```

## 16. 进度表

| 阶段 | 状态 | 下一动作 |
|---|---|---|
| Phase 0 — PyTorch Training Bridge | ✅ 基本完成 | 仅按需回补 |
| Phase 1 — Transformer Foundations | 🔓 进行中 | MHA → RoPE → tensor layout |
| Phase 2 — TinyLM | ⬜ | Gate 1 后开始；C++ Ch00–03 另按 C0 准入 |
| Phase 3 — Inference Foundations | ⬜ | Gate 2 后开始；C++ 执行要求 C0，通过 C1 后升级 |
| Phase 4 — C++ Inference Runtime + Mini SGL 对照 | ⬜ | Gate 3 + C1 后开始 |
| Phase 5 — Read Mini-SGLang | ⬜ | Gate 4 后开始 |
| Phase 6 — Inference Runtime Portfolio | ⬜ | Phase 5 中选题 |
| Phase 7 — Rollout Runtime | ⬜ | Gate 4–6 后开始 |
| Phase 8 — Alignment / RLVR / Agentic RL | ⬜ | Gate 7 后开始 |

附加门槛状态：C0（C++ 工程）待验收，C1（C++ 执行链）未开始。Phase 1–2 中的桥接不代表阶段整体完成。每个 Gate 只在自己的代码/阅读实验、测试或基线对照、中文记录和局限说明齐全后通过。

## 17. 当前立即执行的学习单元

```text
Multi-Head Attention 直觉与 shape
→ qkv projection / split
→ reshape + transpose + stride
→ causal attention toy example
→ 学习者独立实现 MHA
→ correctness / gradient review
→ RoPE 二维旋转与 API toy example
→ full-sequence vs start_pos decode
→ tensor layout 实验与总结
```

本单元结束后再进入 RMSNorm、SwiGLU 和完整 Transformer block，不插入 Triton kernel 支线。

并行 C++ 准备的最短顺序：环境盘点 → 所有权/构建/shape/错误路径练习 → C0 review → Ch00 基线 → view/CPU 算子。基础熟悉的题用已有证据验收，避免重复刷题。当前优先产物是 MHA/RoPE/layout 的个人实现与测试、C++ 工程入门证据；不以本期结束前完成全部引擎为硬目标。

## 18. 资料核对

- [Stanford CS336 Spring 2026](https://cs336.stanford.edu/)：Lecture 10 为 Inference；Assignment 1 是 Basics；Assignment 2 是 Systems；Assignment 5 是 Alignment and Reasoning RL。
- [Datawhale Diy-LLM](https://datawhalechina.github.io/diy-llm/)：Ch10 推理，Ch13 训练流程，Ch14 RLVR。
- [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang)：Part I 推理基础，Part II 构建 Mini SGL，Part III 真实 SGLang 高级技术，Part IV profiling/PR。
- [mini-sglang](https://github.com/sgl-project/mini-sglang)：真实轻量框架，覆盖 Radix Cache、Chunked Prefill、Overlap Scheduling、TP 和 attention backends。

上述课程结构沿用 2026-10-31 核对记录，执行对应阶段前重新核对。

2026-11-13 新增核对（文档及部分源码抽查，未执行完整测试）：

- [cpp-llm-engine USAGE](https://github.com/datawhalechina/cpp-llm-engine/blob/main/USAGE.md)：章节、工具链与运行入口。
- [CPU 六任务计划](https://github.com/datawhalechina/cpp-llm-engine/blob/main/source/guides/cpu_learning_plan.md)：阶段产物与验证记录；不直接视为本期活动任务表。
- [Foundation](https://github.com/datawhalechina/cpp-llm-engine/blob/main/exercises/foundation/README.md)：starter 与参考回归的区别。
- [llm-algo-leetcode 推理路线](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/topic_discussion/inference_optimization/intro.md)：按问题选读。
- [推理对比实验](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/docs/02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md)：机制模拟和真实后端实验分别记录。

实际开始实验时固定 upstream commit，记录资料版本。教程测试只是最低参考；个人测试还需覆盖数值、causal mask、增量生成、资源生命周期和异常路径。未测量的速度、容量或质量收益不得写成结论。
