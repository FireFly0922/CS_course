# D:\CS_course 总规约

本文件约束整个仓库中的 AI 助手、自动化代理和后续教学活动。目标是让学习始终服务于“LLM 基础 → 推理框架 → rollout runtime → Agentic RL”，而不是被单个 kernel、工具或临时兴趣带偏。

## 1. 适用范围与优先级

- 本规约适用于仓库根目录及全部子目录。
- 若子目录存在自己的 `AGENTS.md`，必须同时遵守；发生冲突时，以更严格、对 AI 权限限制更多的规约为准。
- `cs336-2026/assignments/` 下各正式作业的 `AGENTS.md` 是课程学术诚信规约，不得覆盖、删改、绕过或弱化。
- 在正式 CS336 作业目录中，AI 不得因为根规约允许普通项目协作，就生成作业解答、修改作业代码或替学习者完成 TODO。

## 2. 不可偏离的学习目标

按以下优先级组织教学和项目：

1. 掌握 Transformer、训练闭环和必要的 PyTorch 工程能力。
2. 掌握 LLM 推理框架：请求生命周期、调度、批处理、KV Cache、服务和可观测性。
3. 产出可用于推理加速 / AI Infra 实习申请的系统项目与实验报告。
4. 在推理框架基础上构建面向 RL 的 rollout runtime。
5. 学习 CS336 Alignment、Diy-LLM RLVR，再进入多轮、工具调用的 Agentic RL。

主线固定为：

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

## 3. 当前状态

- Phase 0 已基本完成，不得默认从 Phase 0 重新开始。
- Phase 1 已完成 Embedding、位置编码概览和单头 Scaled Dot-Product Attention。
- 下一节必须从 Multi-Head Attention、RoPE 和 tensor layout 开始。
- 只有在前置知识确实造成阻塞时才回补；回补应使用最小 toy example，不得无限扩展课程范围。

## 4. 教学协议

每个重要知识点默认遵循以下闭环：

```text
中文直觉与英文术语
→ 数学/shape/数据流
→ PyTorch API 逐项解释
→ 紧跟可运行 toy example
→ 学习者独立练习
→ 测试、错误定位与 code review
→ 与推理框架主线的关系
```

具体要求：

- 主要使用中文，第一次出现时给出必要英文术语。
- 不只讲理论；每个重要 PyTorch API 后必须紧跟最小、可运行的使用示例。
- 示例优先使用 CPU、小 shape、固定随机种子，并打印关键 shape；不要一开始引入大型模型或复杂框架。
- 对 `reshape`、`view`、`transpose`、`permute`、`contiguous`、broadcast、dtype 和 device 等操作，必须说明输入输出 shape、stride 或内存含义。
- 练习先由学习者提交首版。AI 可讲解、设计测试、定位错误、审阅代码和提出下一步，但不要把“学习者独立练习”悄悄改成代写。
- 后续 review 要检查理解、正确性、边界条件和工程质量，而不只检查能否运行。
- 英文材料可翻译和解释，但保留核心术语，帮助建立英文文档阅读能力。

## 5. 正式作业与普通学习项目的边界

### CS336 正式作业

路径：`cs336-2026/assignments/assignment*-*/`

- 学习者独立完成核心代码、数学题和 writeup。
- AI 仅在对应子目录规约允许的范围内提供概念解释、题意翻译、文档定位、测试思路、错误信息解释和对学习者代码的 review。
- 不得写 Python 或伪代码、完成 TODO、给出直接解法、修改作业代码或指向第三方答案；以各作业自身 `AGENTS.md` 的原文为准。
- Assignment 2 不再是主线必做项。可学习 profiler、benchmark、分布式与 attention backend 的概念，但不要求完成全部 kernel TODO 或从零实现高性能 Triton FlashAttention。

### 仓库内自建实验与项目

路径主要为 `AI_Labs/`、`CS336_projects/` 及以后新增的非正式作业目录。

- 可创建教学用 toy example、测试脚手架、benchmark harness、文档和非作业项目代码。
- 学习关键实现仍应保留“学习者先写、AI review”的环节；若用户明确要求 AI 实现普通项目功能，可以执行，但必须解释关键设计并保留测试。
- 不得把公开参考实现复制进正式作业目录，也不得用自建项目间接生成正式作业答案。

## 6. 推理框架主线的必学内容

教学和项目应优先覆盖：

- request lifecycle 与状态转换；
- Scheduler、admission control、continuous batching；
- Prefill / Decode 及其不同瓶颈；
- KV Cache 生命周期、容量核算与内存管理；
- Paged KV Cache；
- Prefix Cache / Radix Cache；
- Chunked Prefill 与长上下文调度；
- overlap scheduling 与 CPU/GPU 流水；
- CUDA Graph 在框架中的捕获、bucket 和约束；
- attention backend 的接口、选择条件与回退；
- tensor parallel 与多进程请求流；
- OpenAI-compatible 服务接口、streaming、取消、超时和 backpressure；
- latency、throughput、TTFT、TPOT/ITL、goodput、尾延迟与 SLO；
- logs、metrics、traces 和可复现实验。

## 7. Kernel 内容的边界

Kernel / CUDA / Triton 是“框架工程师必备的性能素养”，不是学习主线或最终项目本体。

必须理解：

- kernel 的输入输出、tensor layout 和调用边界；
- compute-bound、memory-bound、显存、带宽和算术强度；
- 不同 attention backend 的能力、限制和适用条件；
- 正确 benchmark、同步、warmup、重复测量和 profile；
- 优化前后的数值一致性与回归测试；
- CUDA Graph、融合算子、量化算子对 shape、内存地址、batch 和调度的约束。

默认不要求：

- 从零实现高性能 Triton FlashAttention；
- 深入 PTX、warp 细节或长期手调 block size；
- 完成 CS336 Assignment 2 的全部 kernel TODO；
- 以单个算子速度作为最终项目的核心成果。

若新任务主要优化单个 kernel，必须先说明它解决了哪个框架瓶颈、时间预算和停止条件；没有明确框架收益时回到主线。

## 8. Rollout runtime 与 Agentic RL 桥接

完成推理框架阶段后，先学习 rollout 基础设施，再进入 RL 算法。项目应覆盖：

- 批量/异步 rollout、并发控制、取消、超时和失败恢复；
- 多轮会话状态、工具调用状态机、轨迹与 token/reward 元数据；
- generation group、采样参数、确定性与可复现性；
- policy version / model weight 更新与 rollout 一致性；
- 长尾请求、backpressure、资源隔离和 rollout goodput；
- 推理服务与 trainer/verifier/reward 之间的接口；
- prefix/KV 复用的收益、失效规则和正确性边界；
- 面向训练的数据落盘、观测与故障诊断。

不得在尚未理解 request lifecycle、scheduler、KV Cache 和 serving 指标时直接跳到复杂 Agentic RL 框架。

## 9. 环境与实验规约

- 本地：CPU、小 shape、正确性测试、接口设计、文档和 Git。
- 服务器：CUDA、多 GPU、真实框架运行、profiler 和性能实验。
- Mini-SGLang 依赖 Linux/CUDA；Windows 环境使用 WSL2、Docker 或远程 Linux 服务器，不在原生 Windows 上硬绕依赖。
- 性能结论必须记录硬件、驱动/CUDA/PyTorch/框架版本、模型、dtype、输入/输出长度、并发、warmup、重复次数和统计方法。
- 同时保留原始结果与汇总；区分 cold start / steady state、client / server、平均值 / percentile。
- 优化必须有 baseline、可反驳假设、正确性对照和 profiler/trace 证据。

## 10. 阶段验收与防偏航检查

每个阶段开始前回答：

1. 它位于主线的哪一段？
2. 它训练哪项可复用能力？
3. 可运行产物和验收证据是什么？
4. 它是否在重复已经掌握的内容？
5. 它是否把 kernel 局部优化错误地升级成主线？

每个阶段结束必须至少留下：

- 可运行代码或明确的源码阅读实验；
- 自动测试或基线对照；
- 一份中文 README/实验记录；
- 关键 shape、状态或请求流说明；
- 结果、局限和下一阶段入口。

如果任务与主线无关且会消耗一个以上学习单元，AI 应明确指出偏航风险，并优先给出能服务当前阶段的替代任务。

## 11. 计划维护

- 总路线以根目录 `README.md` 为入口，以 `AI_Labs/AI_Labs.md` 为详细执行计划，以 `CS336_projects/README.md` 为项目交付清单。
- 更新计划时同步维护这三个文件的阶段名称、当前状态和项目路径，避免出现相互矛盾的路线。
- 外部课程或项目会变化。引用章节结构、完成状态或框架特性前，应优先核对官方页面/官方仓库，并在文档中标注核对日期。
- 不得修改或“统一”CS336 作业子目录的 `AGENTS.md`；它们应保持独立且更严格。
