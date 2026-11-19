# MiniLM 学习规划

对应总路线的第一阶段；后续依次为 cpp-llm-engine、zero-to-sglang、CS106L。总览见 [学习路线](../README.md)。

## 目标与当前进度

亲手实现一个 Decoder-only Transformer，完成预训练和后训练，得到能进行基本中文对话、遵循简单指令，并在有限任务上表现出可测能力的小语言模型。

学习重点是理解模型结构并提高 PyTorch 实现能力：能从公式追踪到张量操作，解释每一步 shape、内存布局和梯度，能独立修改结构、定位训练问题。

PyTorch 基础已基本完成；已学习 Embedding、位置编码概览和单头 Scaled Dot-Product Attention。目录中还已有 MHA、RoPE、TransformerBlock 和模型整合代码；接下来从 MHA、RoPE 和 tensor layout 的理解与现有实现复核开始，再推进完整模型验收，不重复已掌握内容。

## 目录阶段与当前完成情况

下表对应现有目录。P0、P1 是 MiniLM 内部阶段；P1 下的三个编号目录是其学习单元，与总路线中的四个项目阶段分开编号。代码存在仅说明已有实现，不能单独证明测试通过或独立掌握。

| 阶段或单元 | 对应目录 | 主要任务 | 当前完成情况 | 下一步 |
| --- | --- | --- | --- | --- |
| P0：PyTorch 基础 | [P0](P0/) | Tensor、autograd、训练与评估、dtype/device、数据加载 | 基本完成；已有笔记和二分类训练示例 | 仅在后续遇到阻塞时回补 |
| P1：Transformer 基础 | [P1](P1/) | 从模型组件到完整 Decoder-only LM | 进行中；已有组件、Block 和模型整合代码，整体验收未完成 | 复核 MHA、RoPE、layout，再检查整合与训练行为 |

数据与训练底座、正式预训练、后训练仍按下文顺序推进。当前尚无对应的独立阶段目录，暂不虚构 P2 等目录或添加无效链接；建立目录后再补充进度行。

## 现有文件索引

链接以本文件所在的 Mini-LM 目录为起点。以下说明依据文件当前内容，复习模板保留待实现部分，不计为未完成的正式提交。

### P0：PyTorch 基础

| 文件 | 作用 |
| --- | --- |
| [before_P0.md](P0/before_P0.md) | PyTorch 基础笔记，覆盖 Tensor API、shape/内存布局、autograd、loss、优化器、训练评估模式、dtype/device 和 DataLoader。 |
| [P0.py](P0/P0.py) | 人工二分类训练示例，串起数据加载、含 BatchNorm 的网络、反向传播、参数更新和推理，并观察 BatchNorm 状态变化。 |

### P1：阶段笔记

| 文件 | 作用 |
| --- | --- |
| [before_P1.md](P1/before_P1.md) | Transformer 学习笔记，涵盖 Embedding、位置编码、单头与多头注意力、RoPE、Norm、残差、SwiGLU 和 token 到 logits 的前向流程。 |

### P1.1：组件练习

| 文件 | 作用 |
| --- | --- |
| [embedding_Labs.py](P1/01_components/embedding_Labs.py) | Embedding 练习题，要求检查查表、one-hot 对照、padding 梯度和内存占用；当前内容为题目，没有练习实现。 |
| [MHA.py](P1/01_components/MHA.py) | 显式计算多头因果注意力的演示，展示投影、拆头、mask、softmax、合头和输出 shape，适合追踪 attention 数据流。 |

### P1.2：RoPE 与 Transformer Block

| 文件 | 作用 |
| --- | --- |
| [RotaryEmbedding.py](P1/02_block/RotaryEmbedding.py) | RoPE 实现，构建旋转缓存并按起始位置旋转 Q/K；包含独立运行时的 shape 演示。 |
| [Tiny_transformer.py](P1/02_block/Tiny_transformer.py) | TransformerBlock 当前实现，连接两个 RMSNorm、Attention/FFN 和残差分支；文件本身不是完整语言模型。 |
| [before_Tiny_transformer.py](P1/02_block/before_Tiny_transformer.py) | TransformerBlock 练习模板，保留接口、TODO 和占位异常，用于独立复习与重写。 |

### P1.3：完整模型与单步训练

| 文件 | 作用 |
| --- | --- |
| [assembly_lab.md](P1/03_assembly/assembly_lab.md) | 整合练习指南，说明模块关系、接口、A–F 任务及验收边界；部分任务描述尚未随实现更新，不能单凭指南判断代码进度。 |
| [assembly_lab.py](P1/03_assembly/assembly_lab.py) | 当前整合实现，包含配置、SwiGLU、带 RoPE 的 Attention、TinyDecoderLM、标签错位和 train_step；尚未包含完整训练循环、checkpoint 或生成入口。 |
| [before_assembly_lab.py](P1/03_assembly/before_assembly_lab.py) | 按当前接口整理的整合复习模板，保留配置、TODO 和提示，用于独立复做；不是最初实现的逐字历史备份。 |

### 学习路线与指导规约

| 文件 | 作用 |
| --- | --- |
| [README.md](../README.md) | 四阶段学习路线、各阶段目标和资料入口。 |
| [AGENTS.md](../AGENTS.md) | Agent 指导边界、教学方式、各阶段学习目标与完成标准。 |

## 学习顺序

### 1. Transformer 结构与实现

依次学习 MHA、RoPE、tensor layout、RMSNorm、SwiGLU、残差连接和完整 Transformer，随后理解 GQA 等结构变化。从 token 到 logits 逐步实现并验证，掌握各模块的数学含义、shape 和连接方式。

核心前向由学习者实现，可以使用 PyTorch 张量运算、autograd 和基础层；完整模型类或现成 Transformer block 不替代这一学习过程。

### 2. 数据与训练底座

学习 Tokenizer/BPE、数据清洗与划分、packing、target shift，以及 loss、优化器、学习率调度、验证、checkpoint 和生成循环。

先用极小数据验证实现，再建立可恢复、可复现的训练流程。小数据过拟合仅作为底座检查，模型项目继续推进到实际语言能力训练。

### 3. 从随机初始化开始预训练

准备真实语料，完成清洗与去重，并根据 GPU、显存和预算确定模型规模与训练量。先短程试训，再开展正式预训练，持续记录验证表现、生成样例和失败案例。

按需学习混合精度、梯度累积与 DDP 多卡训练；在理解原生 PyTorch 训练循环后，根据显存与吞吐瓶颈选择 FSDP 或 DeepSpeed/ZeRO 做实际训练与对照。理解参数、梯度和优化器状态的分片，以及通信、checkpoint 和恢复流程；实际多卡实验按硬件条件安排，不要求同时完整实现两套接入。

### 4. 后训练与模型评估

从自己的预训练 checkpoint 开始完成 SFT，理解对话模板、多轮数据和 assistant-only loss。随后根据目标、数据、奖励来源和算力，选择合适的后训练实验：

- DPO：利用成对偏好数据学习偏好优化。
- PPO/RLHF：在具备可靠偏好数据与奖励建模条件时，学习奖励模型、策略采样、价值估计、KL 约束与策略更新。
- GRPO/RLVR：对有可靠可验证奖励的任务，学习组内采样、优势估计、奖励设计与策略优化。

RLHF 是利用人类反馈的训练范式，PPO 是可用算法之一；RLVR 按奖励来源区分，GRPO 是可用算法之一。先理解这些关系，再选择实验，不要求把所有方法依次做完。RL 后训练可以在本阶段开展，不以完成后续推理引擎项目为前提。

用独立评测样本检查表达连贯性、指令遵循、短文理解、简单任务和多轮对话。保留 Base、SFT 和所选后训练方法的比较及失败案例；RL 实验同时检查奖励投机、回答长度变化和通用能力退化，不把训练奖励上升直接视为模型能力提升。

多轮工具调用与环境交互的 Agentic RL 暂不安排；普通对话及任务型 RL 后训练不受此限制。

## 资料分工

| 资料 | 用途 |
|---|---|
| [Diy-LLM](https://datawhalechina.github.io/diy-llm/) | 中文知识主线，串起结构、数据、训练与评估 |
| [CS336](https://cs336.stanford.edu/) | 深入数学、实现原理和实验方法；正式作业独立完成 |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | 阅读紧凑的模型与训练循环实现 |
| [MiniMind](https://github.com/jingyaogong/minimind) | 参考小模型预训练、SFT 与后训练的工程流程 |
| [PyTorch FSDP](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) | 学习训练状态分片与分布式训练 |
| [DeepSpeed](https://www.deepspeed.ai/tutorials/zero/) | 学习 ZeRO 与训练系统接入 |

围绕当前问题选读，避免重复完整学习多套材料。

## 实践方式与阶段衔接

每个重要知识点遵循“中文解释 → 数学与 shape → API 可运行小例子 → 学习者独立实现 → 测试与 review”。Agent 负责指导、练习设计、错误分析和审阅，不代写模型、训练代码或练习答案，也不直接修改学习者代码。具体练习、配置、测试和验收细则在 Mini-LM 各学习单元中维护。

本阶段保留一套持续演进的模型代码、训练配置、checkpoint、评估结果和中文记录。完成后进入 cpp-llm-engine，将模型结构与生成流程的理解迁移到 C++ 推理实现；本阶段不并行安排 CS106L 或刷题。
