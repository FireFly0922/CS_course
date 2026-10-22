# math2metal

> 从数学公式到 GPU kernel —— 一条通往 AI Infra 的自学路径。
>
> *From math formulas to GPU kernels — a self-paced road to AI Infra.*

本仓库收录我的 **CS224N / CS336 作业实现**，以及按 [AI_Labs.md](./AI_Labs.md) 计划推进的 **分层 coding labs**。

目标不是"刷完两门课"，而是把深度学习**往下学**：每个公式都落到 tensor、内存和 kernel 上，最终能够独立阅读、实现、调试和 profiling 一个小型语言模型的训练 / 推理系统，为进入 **AI Infra（训练 & 推理加速）** 方向打基础。

---

## 核心方法论：一条链

AI_Labs 中每一道题都沿着同一条链推进：

```text
数学公式
  → 算法步骤
  → Tensor shape
  → Flat memory / stride
  → Operator
  → Kernel
  → Hardware
```

这条从 **math** 一路下探到 **metal**（裸机 / 硬件）的链，就是仓库名字的由来。

## 路线分工

- **CS224N**：补齐神经网络、序列模型、Attention、Transformer 和 PyTorch 基础，提升 coding 能力。
- **CS336**：在已有模型基础上，学习语言模型训练、系统优化、GPU kernel、分布式训练、推理和数据工程。
- **AI_Labs.md**：把两门课的知识点拆成 28 个可验收的小实验，作为从"补基础"过渡到"真实 AI Infra 工程"的桥。

## 仓库结构

```text
.
├── README.md
├── AI_Labs.md              # 训练计划总纲：lab 定义、验收标准与详细完成记录
├── cs224n/
│   ├── assignment1/
│   ├── assignment2/        # word2vec 推导 + 神经网络基础
│   └── ...
├── cs336/
│   └── assignment1/        # 从零构建语言模型
└── labs/                   # 与 AI_Labs.md 中 Lab 01–28 一一对应
    ├── lab01-mlp-cpp/
    ├── lab02-gradcheck-cpp/
    └── ...
```

每个 lab 目录按统一的交付物标准组织：

```text
labXX-<name>/
├── src/                    # 源代码
├── build.sh / run.sh       # 编译 / 运行命令，可复现
├── test/                   # 固定测试输入与期望输出
└── NOTES.md                # shape 说明、复杂度分析、验收自检、与真实系统的差距
```

## 路线图

| 阶段 | 内容 | 交付能力 |
| --- | --- | --- |
| CS224N 作业 | Assignment 1–5 | 作业驱动的公式推导与 PyTorch 实现 |
| Part I · Lab 01–05 | C++/NumPy 手写 MLP、梯度检查器、embedding 表、相似度引擎 | 把公式写成 flat-memory 上的算子 |
| Part II · Lab 06–13 | RNN / BPTT、Attention、tensor layout、Tiny Transformer | 理解序列模型与 Transformer 数据流 |
| Part III · Lab 14–17 | Mini-GPT 独立重建、tokenizer、可靠训练循环、FLOPs / 显存核算 | 独立组织一个小型语言模型训练项目 |
| Part IV · Lab 18–23 | Profiler、Triton、tiled attention、DDP、KV cache、`torch.compile` | 从模型代码追到 kernel 与硬件（AI Infra 核心） |
| Part V · Lab 24–28（可选） | Scaling law、数据 pipeline、LoRA、SFT / preference loss、toy RLVR | 后训练与数据工程扩展 |

### 优先级主线

```text
当前必须完成:   Assignment 2 → Lab 01 → Lab 02 → Lab 03
CS224N 基础主线: Lab 04 → 06 → 07 → 08 → 09 → 11 → 12 → 13
CS336 核心主线:  Lab 14 → 15 → 16 → 17 → 18 → 19 → 20 → 21 → 22 → 23
可选扩展:        Lab 24 → 25 → 26 → 27 → 28
```

## 使用规则（摘录自 AI_Labs.md）

- **开启条件**：知识点已经学过，且前置题目已经通过；以知识点为准，不机械依赖课号。
- **每道题的标准交付物**：源代码、编译或运行命令、固定测试输入与运行结果、核心变量 shape 与含义、时间 / 空间复杂度和主要内存访问说明、按验收标准逐项自检。
- **完成定义**：程序可运行且测试输入可复现；能解释每个核心变量的 shape、含义和生命周期；能说明复杂度；至少有一种独立正确性检查（手算 / 对照实现 / 有限差分 / 框架结果）；能指出当前实现与真实训练 / 推理系统的差距。
- **AI 使用边界**：第一版实现不让 AI 直接生成完整代码；可询问公式与概念、接口与数据结构设计、错误定位、测试方案和 code review。

## 进度追踪

- CS224N 进度：第 4 节课附近
- 进行中：Assignment 2
- 已开启：Lab 01

状态图例：`⬜ 未开启` · `🔓 已开启` · `🚧 进行中` · `✅ 已完成`

代码路径、验收自检等详细记录见 [AI_Labs.md](./AI_Labs.md) 对应条目的「完成记录」，两张表保持同步更新。

### CS224N 作业

| 编号 | 名称 | 状态 | 完成日期 |
| --- | --- | --- | --- |
| Assignment 1 | word2vec | ⬜ | — |
| Assignment 2 | word2vec 推导 + 神经网络基础 | 🚧 | — |
| Assignment 3 | 依存解析 | ⬜ | — |
| Assignment 4 | 机器翻译 / RNN | ⬜ | — |
| Assignment 5 | Transformer / 预训练 | ⬜ | — |

### Part I · 神经网络与反向传播基础（Lab 01–05）

| 编号 | 名称 | 状态 | 完成日期 |
| --- | --- | --- | --- |
| Lab 01 | C++ 两层 MLP Forward / Backward | 🔓 | — |
| Lab 02 | C++ 梯度检查器 | ⬜ | — |
| Lab 03 | 手动反传 vs PyTorch Autograd | ⬜ | — |
| Lab 04 | C++ Embedding Table | ⬜ | — |
| Lab 05 | C++ 词相似度引擎 | ⬜ | — |

### Part II · 序列模型、Attention 与 Transformer（Lab 06–13）

| 编号 | 名称 | 状态 | 完成日期 |
| --- | --- | --- | --- |
| Lab 06 | NumPy Vanilla RNN Forward | ⬜ | — |
| Lab 07 | NumPy BPTT | ⬜ | — |
| Lab 08 | NumPy Toy Attention | ⬜ | — |
| Lab 09 | C++ 单头 Causal Attention | ⬜ | — |
| Lab 10 | Streaming Attention | ⬜ | — |
| Lab 11 | Multi-Head Layout | ⬜ | — |
| Lab 12 | Tensor = Storage + Shape + Stride | ⬜ | — |
| Lab 13 | Tiny Transformer / Mini LM | ⬜ | — |

### Part III · CS336 过渡（Lab 14–17）

| 编号 | 名称 | 状态 | 完成日期 |
| --- | --- | --- | --- |
| Lab 14 | Mini-GPT 独立重建 | ⬜ | — |
| Lab 15 | BPE Tokenizer 与 Dataset Packing | ⬜ | — |
| Lab 16 | 可靠训练循环 | ⬜ | — |
| Lab 17 | Resource Accounting：FLOPs / 显存 / AI | ⬜ | — |

### Part IV · CS336 Systems（Lab 18–23）

| 编号 | 名称 | 状态 | 完成日期 |
| --- | --- | --- | --- |
| Lab 18 | GPU Benchmark 与 PyTorch Profiler | ⬜ | — |
| Lab 19 | Triton Kernel 入门 | ⬜ | — |
| Lab 20 | Tiled / Memory-Efficient Attention | ⬜ | — |
| Lab 21 | DDP 最小实验 | ⬜ | — |
| Lab 22 | KV Cache 推理实验 | ⬜ | — |
| Lab 23 | `torch.compile` 优化实验 | ⬜ | — |

### Part V · Scaling、Data 与 Post-training 扩展（Lab 24–28，可选）

| 编号 | 名称 | 状态 | 完成日期 |
| --- | --- | --- | --- |
| Lab 24 | Scaling Law 小实验 | ⬜ | — |
| Lab 25 | 数据过滤、去重与 Packing Pipeline | ⬜ | — |
| Lab 26 | LoRA Linear | ⬜ | — |
| Lab 27 | Toy SFT / Preference Loss | ⬜ | — |
| Lab 28 | Toy RLVR 闭环 | ⬜ | — |

### CS336 作业

| 编号 | 名称 | 状态 | 完成日期 |
| --- | --- | --- | --- |
| Assignment 1 | Basics：从零构建语言模型 | ⬜ | — |
| Assignment 2 | Systems | ⬜ | — |
| Assignment 3 | Scaling | ⬜ | — |
| Assignment 4 | Data | ⬜ | — |
| Assignment 5 | Alignment & Reasoning RL | ⬜ | — |

> CS224N / CS336 作业主题以当年课程发布为准；CS336 五个方向对应 2026 官方课程结构。

## 完成这条路线后的能力目标

- 独立解释反向传播、RNN、Attention 和 Transformer 的核心数据流；
- 用 C++ flat memory 写出小型算子，用 NumPy 验证公式与 shape；
- 独立组织一个小型语言模型训练项目（checkpoint、gradient accumulation、mixed precision、评估）；
- 用 benchmark 和 profiler 定位性能瓶颈，理解 kernel、显存访问、算子融合、Triton、DDP 和 KV cache 的关系；
- 从"模型代码"进一步追踪到"算子、kernel 和硬件"，把学习重点从补基础转向真实 AI Infra 工程问题。

## 相关资源

- [CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)
- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [nanoGPT](https://github.com/karpathy/nanoGPT)（Lab 14 的参照项目，独立重建、不复制）
- [Triton](https://triton-lang.org/)（Lab 19 起使用）
