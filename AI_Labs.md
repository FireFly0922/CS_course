# CS224N → CS336 基础与 AI Infra Coding Labs

> 这份文档是我的分阶段 coding 训练计划。
>
> CS224N 的定位：补充神经网络、序列模型、Attention、Transformer 和 PyTorch 基础，提升代码能力。
>
> CS336 的定位：在已有模型基础上，继续学习语言模型训练、系统优化、GPU kernel、分布式训练、推理和数据工程。
>
> 最终目标：能够独立阅读、实现、调试和 profiling 一个小型语言模型训练/推理系统，为后续 AI Infra、模型蒸馏、推理优化和后训练项目打基础。

## 1. 这条路线不追求什么

这不是一条“把自己训练成专业 NLP 研究员”的路线，也不要求现在就掌握完整的语言学、信息抽取或 NLP 任务体系。

学习 CS224N 的主要原因是：

- 通过神经网络和 Transformer 训练代码补齐深度学习基础。
- 把公式映射成 tensor、内存和循环，提升 coding 能力。
- 理解 PyTorch 自动求导、训练循环和模型结构，而不是只会调用 API。
- 为 CS336 的大量 Python 实现、GPU 优化和分布式训练做准备。
- 最终服务于 AI Infra、模型压缩/蒸馏、推理部署和后训练等方向。

## 2. 学习主线

```text
CS224N 数学与模型基础
        ↓
C++ / NumPy 手写核心算子
        ↓
PyTorch 复现训练流程
        ↓
Tiny Transformer / Mini-GPT
        ↓
CS336：训练系统、GPU、Triton、并行、推理
        ↓
Profiling、模型优化、蒸馏和后训练项目
```

每道题都沿着下面这条链推进：

```text
数学公式
  → 算法步骤
  → Tensor shape
  → Flat memory / stride
  → Operator
  → Kernel
  → Hardware
```

## 3. 当前状态

- 当前课程进度：CS224N 第 4 节课附近。
- 当前主要任务：完成 Assignment 2。
- 当前只开启：`Lab 01`。
- 当前执行顺序：

```text
Assignment 2
    ↓
Lab 01：C++ 两层神经网络 Forward / Backward
    ↓
Lab 02：C++ Gradient Checker
```

后面的题目只是路线索引，不是当前要同时完成的任务。

## 4. 新增研究小项目：观察模型如何学会泛化

### 先用一句话说明

这个小项目不要求你现在立刻开始，也不是另开一门课。它会在你完成 Tiny Transformer 以后，利用你自己写的模型和训练程序，记录模型在训练过程中每一层的中间结果，研究下面这个问题：

> **模型从“只记住训练题”变成“能够解决没见过的新题”时，内部的数据空间发生了什么变化？模型变大、数据变多或训练计算量增加，是否会以不同方式帮助这种变化发生？**

这里的“潜空间”可以先简单理解为：模型把输入变成一串数字向量后的空间。每个输入都会变成空间中的一个点。模型学习的过程，就是不断调整这个空间以及点的位置，使有共同规律的数据更容易被同一种方式处理。

例如，训练初期模型可能只是记住了训练集中出现过的题目；训练更久以后，它可能学会题目背后的规律，因此能够解决训练中没有出现过的题目。这个项目就是用数据和实验去观察这两个阶段的差别，而不是只看一条 loss 曲线。

### 这个项目最终要做出什么

你将完成一个小型、可复现实验系统：

```text
自己写的 Tiny Transformer
        ↓
在训练过程中定期保存中间表示
        ↓
测量表示空间的结构变化
        ↓
比较训练集、普通测试集和变形后的测试集
        ↓
研究模型规模、数据量、训练计算量与泛化的关系
```

第一版只使用很小的人工任务，例如模运算或简单的组合规律。这样你可以明确知道哪些题目见过、哪些题目没见过，也能把“记忆”和“学会规律”区分开。

### 为什么它适合放进当前路线

- `Lab 13` 提供可以观察的 Tiny Transformer。
- `Lab 16` 提供定期保存模型和继续训练的能力。
- `Lab 17–18` 提供资源统计和性能测量能力。
- `Lab 24` 提供模型规模、数据量和训练预算的对照实验。

因此，研究项目不是与主线并列的大项目，而是把已有实验逐步升级成一个小型研究系统。当前阶段不做它，先完成当前已经开启的题目。

### 文档中的术语约定

为了便于阅读，本文尽量使用中文，并在第一次出现时注明常见英文名：

| 本文用语 | 含义 |
|---|---|
| 中间表示 | 模型某一层输出的数字向量，也常叫 hidden representation |
| 潜空间 | 所有中间表示组成的数字空间，也常叫 latent space |
| 空间结构 | 这些点之间的距离、邻居关系和维度特征，也常叫 representation geometry |
| 泛化 | 模型处理训练时没有见过、但遵循相同规律的数据的能力 |
| 分布外泛化 | 测试数据与训练数据存在有意变化时仍然正确，也常写作 OOD generalization |
| 规模扩展 | 改变模型参数量、数据量或训练计算量，也常叫 scaling |
| 检查点 | 训练过程中的模型保存文件，也常叫 checkpoint |

这部分研究只需要在确实用到某个概念时学习它，不要求一开始记住所有术语。

## 5. 使用规则

### 4.1 开启方式

每道题按照“知识点已经学过，并且前置题目已经通过”后再开启。具体课程周次可能因 CS224N/CS336 版本不同而变化，因此以知识点为准，不机械依赖课号。

### 4.2 每道题的标准交付物

1. 源代码。
2. 编译或运行命令。
3. 固定测试输入和运行结果。
4. 核心变量的 shape 与含义。
5. 时间复杂度、空间复杂度和主要内存访问说明。
6. 按本题验收标准逐项自检。

### 4.3 AI 使用边界

第一版实现不要让 AI 直接生成完整代码。可以询问：

- 公式和概念。
- 接口与数据结构设计。
- 编译错误和运行错误定位。
- 测试方案。
- 对自己代码的 review。

代码完成后提交给我，我会重点检查数学正确性、shape、索引、复杂度、数值稳定性、内存布局和实现质量。

### 4.4 通用完成定义

一道题只有同时满足以下条件，才算完成：

- 程序能够运行，并且测试输入可复现。
- 能解释每个核心变量的 shape、含义和生命周期。
- 能说明时间复杂度、空间复杂度和主要数据访问。
- 至少有一种独立正确性检查：手算、对照实现、有限差分或框架结果。
- 能指出当前实现与真实训练/推理系统之间的差距。

---

# Part I：CS224N 神经网络与反向传播基础

这一部分的目标不是深入 NLP 任务，而是借助 CS224N 的神经网络内容建立“公式 → 代码”的基本能力。

## Lab 01 — C++ 两层神经网络 Forward / Backward

### 开启条件

- 学完反向传播、链式法则和基本神经网络。
- Assignment 2 已经涉及 tensor derivative 或神经网络基础。

### 题干

使用纯 C++17 实现一个单样本、二分类的两层 MLP，并手动完成 forward 和 backward。

固定计算图为：

$$
x \rightarrow z_1=W_1x+b_1 \rightarrow h=ReLU(z_1)
\rightarrow z_2=W_2h+b_2 \rightarrow p=softmax(z_2)
\rightarrow L=-\log p_y
$$

固定维度为：

$$
x\in\mathbb{R}^2,\quad W_1\in\mathbb{R}^{3\times2},\quad
W_2\in\mathbb{R}^{2\times3}
$$

程序必须保存 `x`、`z1`、`h`、`z2`、`prob`、`loss`，并计算：

$$
\frac{\partial L}{\partial W_1},\quad
\frac{\partial L}{\partial b_1},\quad
\frac{\partial L}{\partial W_2},\quad
\frac{\partial L}{\partial b_2},\quad
\frac{\partial L}{\partial x}
$$

### 限制

- 只允许 C++17 标准库。
- 建议使用 `vector<float>` 或 `vector<double>`。
- 禁止 PyTorch、Eigen、BLAS、Armadillo 和任何 autograd 库。
- 只处理一个样本，不做 batch 和训练循环。
- softmax 必须使用减去最大值的稳定实现。

### 目标

- 将计算图中的数学变量对应到缓存和梯度数组。
- 理解 forward activation 为什么必须在 backward 前保存。
- 能从输出层沿计算图反向推导到输入。
- 区分参数、激活值和梯度。

### 验收标准

- [ ] 程序能编译运行，且无越界访问。
- [ ] ReLU 在 `z1 <= 0` 时的梯度为 0。
- [ ] softmax 概率非负，概率和接近 1。
- [ ] `dL/dz2 = prob - one_hot(label)`。
- [ ] 每个梯度的 shape 与对应参数一致。
- [ ] 能写出本题时间复杂度和空间复杂度。

### 完成记录

- 状态：`未开始`
- 完成日期：`待填写`
- 代码路径：`待填写`

---

## Lab 02 — C++ Gradient Checker

### 开启条件

Lab 01 的 forward 和解析梯度已经通过基础人工检查。

### 题干

为 Lab 01 增加有限差分梯度检查。对每个参数 `theta_i`，计算数值梯度：

$$
\frac{\partial L}{\partial \theta_i}
\approx
\frac{L(\theta_i+\epsilon)-L(\theta_i-\epsilon)}{2\epsilon}
$$

将数值梯度与手写 backward 的解析梯度逐项比较。

### 限制

- 使用中心差分。
- 至少测试两组参数和两组标签。
- 测试参数不要故意让 ReLU pre-activation 等于 0。
- 不允许修改解析 backward 来适配数值梯度。

### 目标

- 理解底层算子不仅要能运行，还要能够验证正确性。
- 区分解析梯度错误、浮点误差和不可导点问题。
- 建立以后检查自定义 PyTorch、CUDA 或 Triton backward 的基础方法。

### 验收标准

- [ ] 输出解析梯度、数值梯度、绝对误差和相对误差。
- [ ] 正常测试下误差达到你设定的阈值，并能解释阈值。
- [ ] 至少构造一个故意错误的 backward 版本，证明 checker 能发现错误。
- [ ] 能解释 `epsilon` 太大或太小分别会造成什么问题。

---

## Lab 03 — 手动反传 vs PyTorch Autograd

### 开启条件

完成 Lab 01–02，并能使用 PyTorch 创建 tensor、计算 loss 和读取 `.grad`。

### 题干

在 PyTorch 中复现 Lab 01 的 MLP。使用完全相同的输入、参数和标签，分别获得：

1. 手动公式计算的梯度。
2. PyTorch autograd 计算的梯度。

逐参数比较两者的 loss 和梯度。

### 目标

- 理解 autograd 是如何记录运算关系并执行反向传播的。
- 理解 leaf tensor、`requires_grad`、`.grad` 和计算图。
- 为阅读训练框架和自定义 backward 做准备。

### 验收标准

- [ ] 手动梯度与 autograd 梯度接近。
- [ ] 能指出每个 forward 中间结果对应的反向依赖。
- [ ] 能解释比较前为什么需要清空旧梯度。
- [ ] 能解释 `torch.no_grad()` 与训练阶段的区别。

---

## Lab 04 — C++ Embedding Table

### 开启条件

学过词向量、token id 与 embedding lookup。

### 题干

使用连续的一维数组保存大小为 `V × D` 的 embedding table。给定 token id，返回对应 embedding；再定义一个简单 loss，只对被查到的行累积梯度。

索引必须显式体现：

```text
table[token_id * D + dim]
```

### 目标

- 理解 embedding lookup 本质上是索引和内存访问，而不是普通矩阵乘法。
- 理解同一个 token 多次出现时梯度为什么要累加。
- 建立 storage、shape、stride 的初步认识。

### 验收标准

- [ ] 正确处理多个 token id。
- [ ] 正确处理重复 token。
- [ ] 对非法 token id 给出明确错误处理。
- [ ] 能写出 lookup 和梯度累加的复杂度。

---

## Lab 05 — C++ Word Similarity Engine

### 开启条件

完成词向量相关内容或 Assignment 1 的词向量部分。

### 题干

给定词表和 embedding，输入一个查询词，计算它与其他词的 cosine similarity，并返回 top-k 相似词：

$$
cos(u,v)=\frac{u\cdot v}{\|u\|\|v\|}
$$

### 目标

- 将 embedding lookup、向量点积、范数和 top-k 组合成小型推理程序。
- 认识算法复杂度和数据访问模式是两个不同问题。

### 验收标准

- [ ] 正确处理零向量和查询词自身。
- [ ] top-k 结果按相似度排序。
- [ ] 能说明暴力扫描的时间复杂度和主要内存开销。
- [ ] 至少使用一组可以人工判断的 embedding 测试。

---

# Part II：CS224N 序列模型、Attention 与 Transformer

这一部分完成后，你应当能理解 Tiny Transformer 的数据流，但不要求把 CS224N 所有 NLP 任务都做成研究项目。

## Lab 06 — NumPy Vanilla RNN Forward

### 开启条件

学完 Vanilla RNN 的 forward 公式。

### 题干

不用 `nn.RNN`，使用 NumPy 实现长度为 4 的 RNN：

$$
h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b_h)
$$

保存每个时间步的 `x_t`、pre-activation 和 `h_t`。

### 验收标准

- [ ] 输出每个时间步的 hidden state。
- [ ] 同一组参数被所有时间步复用。
- [ ] 能说明序列长度增加时参数量和计算量如何变化。
- [ ] 使用小规模输入进行手工检查。

---

## Lab 07 — NumPy BPTT

### 开启条件

理解 RNN 的时间反向传播和梯度跨时间累积。

### 题干

在 Lab 06 上增加一个只依赖最后 hidden state 的 loss，并手动实现长度为 3 的 BPTT。计算 `Wxh`、`Whh`、`bh`、输入和 hidden state 的梯度。

### 目标

- 理解 `Whh` 的梯度为什么来自多个时间步之和。
- 观察梯度跨时间连乘导致的 vanishing/exploding gradient。

### 验收标准

- [ ] 反向传播顺序是从最后时间步到第一个时间步。
- [ ] 能解释 `dh_next` 的来源和用途。
- [ ] 使用有限差分验证至少一个参数矩阵的梯度。
- [ ] 能指出梯度消失/爆炸对应的数学位置。

---

## Lab 08 — NumPy Toy Attention

### 开启条件

学完 Attention 基本公式，尚未要求实现完整 Transformer。

### 题干

给定 `Q`、`K`、`V`，使用 NumPy 实现单头 attention：

$$
S=\frac{QK^T}{\sqrt{d_k}},\quad
P=softmax(S),\quad Y=PV
$$

先实现非 causal 版本，只处理一个 batch、一个 head。

### 验收标准

- [ ] 打印并解释 `QK^T`、`P`、`Y` 的 shape。
- [ ] softmax 沿最后一个维度计算。
- [ ] 每一行概率和接近 1。
- [ ] 能解释 `Q`、`K`、`V` 的作用。

---

## Lab 09 — C++ Single-Head Causal Attention

### 开启条件

Lab 08 正确，并学到 self-attention 或 causal mask。

### 题干

使用纯 C++17、`vector<float>` 和 row-major flat memory，实现：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
S=\frac{QK^T}{\sqrt d}
$$

对 `j > i` 的位置施加 causal mask，再做稳定 softmax，最后计算 `Y=PV`。

固定测试：`T=3, d=2`，`X=[[1,0],[0,1],[1,1]]`，并令 `WQ=WK=WV=I`。

### 限制

- 禁止 Eigen、BLAS、OpenCV、PyTorch 和矩阵库。
- 必须自己实现矩阵乘法、mask 和 row-wise softmax。
- 第一版必须打印中间结果。

### 验收标准

- [ ] `X[i*d+j]` 索引正确。
- [ ] mask 后未来位置的概率为 0 或接近 0。
- [ ] softmax 不因大数输入溢出。
- [ ] 能解释复杂度 `O(T²d)` 与 attention matrix 的 `O(T²)` 内存。
- [ ] 与 Lab 08 相同输入时结果一致。

---

## Lab 10 — Streaming Attention

### 开启条件

Lab 09 正确，并理解 attention score matrix 的内存开销。

### 题干

重写 Lab 09：按输出行 `i` 计算 score、稳定 softmax 和 `P_iV`，计算完成后立即丢弃当前行，不保存完整的 `T×T` score 或 probability matrix。

### 目标

- 将额外空间从 `O(T²)` 降低到 `O(T)`。
- 理解避免中间结果反复写入大内存这一类系统优化思想。
- 区分“降低内存”与“已经实现 FlashAttention”。

### 验收标准

- [ ] 输出与 Lab 09 在误差范围内一致。
- [ ] 源码中不存在完整 `T×T` probability matrix。
- [ ] 能列出当前实现仍未解决的问题，例如重复读取 `K/V`、CPU 循环和缺少 tiling。
- [ ] 能说明时间复杂度为什么仍近似为 `O(T²d)`。

---

## Lab 11 — Multi-Head Layout Lab

### 开启条件

学完 Multi-Head Attention，理解 `d = H × Dh`。

### 题干

使用 NumPy 实现多头 attention，显式处理 `[T,H,Dh]` 的逻辑布局。完成 reshape、按 head 计算 attention、拼接回 `[T,d]`。

### 验收标准

- [ ] 明确写出每个数组的 shape。
- [ ] 能用索引公式解释 `token * d + head * dh + dim`。
- [ ] `H=1` 时与 Lab 08 一致。
- [ ] 能解释 head 拆分与参数矩阵形状的关系。

---

## Lab 12 — Tensor = Storage + Shape + Stride

### 开启条件

开始使用 PyTorch，并学过 `view`、`reshape`、`transpose`、`contiguous`。

### 题干

建立一个 PyTorch notebook，依次对二维 tensor 执行 `view`、`transpose`、`reshape`、`contiguous`，打印 `shape`、`stride`、`numel`、`storage_offset`，并判断是否共享底层 storage。

### 验收标准

- [ ] 能区分 view 操作与复制操作。
- [ ] 能解释 transpose 为什么通常只改变 stride。
- [ ] 能解释非 contiguous tensor 为什么可能不能直接 view 成目标形状。
- [ ] 对每一步给出“是否复制、为什么”的结论。

---

## Lab 13 — Tiny Transformer / Mini Language Model

### 开启条件

学完 Transformer 和 next-token prediction；Assignment 2 已完成或不再占用主要精力。

### 题干

使用 PyTorch 实现一个极小语言模型：token embedding、position embedding、一个 causal self-attention block、MLP、LM head 和 cross-entropy loss。使用极小文本训练，使 loss 明显下降，并生成短文本。

### 限制

- 不使用 HuggingFace Trainer。
- 第一版 attention 不使用高阶封装，至少自己明确构造 `Q`、`K`、`V`、mask 和 loss。
- 模型规模足够小，重点是理解数据流而不是追求效果。

### 验收标准

- [ ] 能解释 `[B,T] → [B,T,C] → [B,T,V]` 的数据流。
- [ ] causal mask 正确。
- [ ] 固定数据上的 training loss 下降。
- [ ] 能解释 logits、target shift 和 cross-entropy 的 shape。

---

# Part III：CS336 过渡阶段

CS336 官方课程是从零构建语言模型的实现型课程，内容覆盖数据、Transformer、训练、评估和部署；官方 2026 课程将作业分为 Basics、Systems、Scaling、Data、Alignment and Reasoning RL 五个方向。

参考：[Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)。

这里不直接要求完整复刻 CS336，而是先用小题补齐进入 CS336 所需的 Python、PyTorch 和系统能力。

## Lab 14 — Mini-GPT 独立重建

### 开启条件

Lab 13 正确，并已经能独立阅读一个小型 Transformer 项目的 `model.py` 和 `train.py`。

### 题干

新建一个不依赖原项目代码的目录，只包含：

```text
mini-gpt/
├── model.py
├── train.py
└── generate.py
```

自己实现 embedding、causal attention、MLP、Transformer block、LM head、训练循环和生成。允许查 PyTorch API，但不要复制 nanoGPT 或其他实现。

### 目标

- 脱离课程作业代码独立搭建最小语言模型。
- 把模型、训练和生成拆成清晰模块。
- 为 CS336 Assignment 1 的完整实现做准备。

### 验收标准

- [ ] 不查看原项目代码也能完成主体实现。
- [ ] Tiny Shakespeare 或自定义小文本上的 loss 下降。
- [ ] 能从输入 token 一直解释到生成 token 的完整路径。
- [ ] 能解释训练时的全序列计算和生成时的逐 token 计算差异。

---

## Lab 15 — BPE Tokenizer 与 Dataset Packing

### 开启条件

开始 CS336 Assignment 1，理解 tokenization、语言模型输入和 target shift。

### 题干

实现一个小型 tokenizer 实验：读取文本，建立 token 到 id 的映射，并将 token 序列切分为固定长度的训练样本：

```text
tokens:  t0 t1 t2 t3 t4 t5 ...
input:   t0 t1 t2 t3
target:  t1 t2 t3 t4
```

可以先完成字符级或词级 tokenizer，再扩展到简化 BPE；重点是理解词表、编码、解码和数据 packing。

### 验收标准

- [ ] `encode` 与 `decode` 在可表示文本上能够互相验证。
- [ ] input 与 target 正确错开一个 token。
- [ ] 正确处理序列不足、边界和未知 token。
- [ ] 能统计词表大小、平均序列长度和数据占用。

---

## Lab 16 — 可靠训练循环

### 开启条件

Lab 14 的模型能够训练，并已经学习 AdamW、学习率 schedule 或 checkpoint 基础。

### 题干

为 Mini-GPT 写一个可恢复的训练循环，至少支持：

```text
取 batch
↓
forward
↓
loss
↓
backward
↓
gradient accumulation
↓
gradient clipping
↓
optimizer.step()
↓
learning-rate schedule
↓
evaluation
↓
checkpoint
```

checkpoint 必须保存模型参数、optimizer 状态、学习率状态、当前 step 和随机数状态中的必要部分。

### 验收标准

- [ ] 能从 checkpoint 继续训练。
- [ ] gradient accumulation 后的结果与等效大 batch 结果基本一致。
- [ ] 梯度裁剪前后能打印梯度范数。
- [ ] 中断并恢复后日志中能看出训练没有从 step 0 重新开始。
- [ ] 能解释 optimizer state 为什么会显著增加显存。

---

## Lab 17 — Resource Accounting：FLOPs、Memory、Arithmetic Intensity

### 开启条件

开始 CS336 的系统优化内容，已经理解矩阵乘法、Attention 和 GPU 基本概念。

### 题干

对 Mini-GPT 的 embedding、Linear、MLP 和 attention 做静态资源估算，并与实际运行数据比较。至少记录参数量、激活数量、估计 FLOPs、输入输出字节数和显存峰值。

对一个算子估计 arithmetic intensity：

$$
AI=\frac{\text{FLOPs}}{\text{Bytes moved}}
$$

### 目标

- 理解参数量、FLOPs、显存和实际 latency 不是同一个指标。
- 初步判断一个算子更可能受计算吞吐还是内存带宽限制。

### 验收标准

- [ ] 资源估算公式写清楚，维度没有混淆。
- [ ] 至少比较两个不同 shape 的算子。
- [ ] 实测结果与估算不一致时，能提出合理原因，例如 kernel launch、cache、框架开销或并行度。
- [ ] 能解释 arithmetic intensity 的意义。

---

# Part IV：CS336 Systems

这一部分是从“会实现模型”进入“让模型高效运行”。它与 AI Infra 目标最直接相关。

## Lab 18 — GPU Benchmark 与 PyTorch Profiler

### 开启条件

有可用 GPU，并完成 Lab 13 或 Lab 14。

### 题干

对 Linear、MLP 和 Attention 测量 latency、吞吐量和显存，比较不同 batch size、sequence length 和 dtype。使用 warmup，并在 GPU 计时前后正确 synchronize。再用 PyTorch Profiler 分析一个 training iteration。

### 验收标准

- [ ] 不只测一次，报告 warmup、重复次数和统计方式。
- [ ] 区分 CPU wall-clock、GPU kernel 时间和端到端时间。
- [ ] 记录输入 shape、dtype 和显存峰值。
- [ ] 分离 data loading、forward、backward 和 optimizer step。
- [ ] 找到至少一个性能瓶颈并提出可验证的优化假设。

---

## Lab 19 — Triton Kernel 入门

### 开启条件

完成 Lab 18，并理解 block、thread、GPU memory hierarchy 和 kernel launch。

### 题干

使用 Triton 实现一个简单算子，例如 vector add、row-wise softmax 或 elementwise activation，并与 PyTorch eager 实现比较结果和性能。

第一版不实现完整 FlashAttention，重点是理解：

```text
Python tensor
    ↓
Triton program
    ↓
block / pointer arithmetic
    ↓
GPU kernel
```

### 验收标准

- [ ] 与 PyTorch 结果在允许误差内一致。
- [ ] 能解释 program id、block size、mask 和指针偏移。
- [ ] 对不同输入规模进行 benchmark。
- [ ] 说明什么时候自定义 kernel 反而可能更慢。

---

## Lab 20 — Tiled / Memory-Efficient Attention

### 开启条件

完成 Lab 10、18、19，并理解 attention 中间矩阵的内存问题。

### 题干

将 Lab 10 的 streaming attention 扩展为分块计算：按 query block 和 key/value block 处理数据，避免显式保存完整 `T×T` attention matrix。先完成 forward，再分析是否能够保持 softmax 的数值稳定性。

### 目标

- 理解 tiling、片上存储和减少 HBM 往返的思想。
- 为阅读 FlashAttention 类实现做准备。
- 区分算法等价、内存复杂度和 kernel 性能。

### 验收标准

- [ ] 与朴素 attention 输出一致。
- [ ] 峰值中间矩阵内存不随 `T²` 增长。
- [ ] 能解释分块 softmax 为什么需要维护行最大值和归一化因子。
- [ ] 能指出当前实现与成熟 FlashAttention kernel 的差距。

---

## Lab 21 — DDP 最小实验

### 开启条件

理解单 GPU training loop、GPU benchmark 和基本进程/通信概念。

### 题干

使用 PyTorch DistributedDataParallel 让两个进程训练同一个极小模型。记录每个进程的 rank、设备、数据分片和梯度同步过程，并比较单 GPU 与 DDP 的吞吐量。

### 验收标准

- [ ] 使用 `torchrun` 启动多进程训练。
- [ ] 每个进程读取不同数据分片。
- [ ] 能解释梯度 all-reduce 发生在哪里。
- [ ] 正确处理只允许主进程打印和保存 checkpoint 的逻辑。
- [ ] 能解释小模型上 DDP 可能没有加速的原因。

---

## Lab 22 — KV Cache 推理实验

### 开启条件

理解 causal attention、逐 token generation 和 tensor layout。

### 题干

为 Mini-GPT 实现两种生成方式：

1. 每生成一个 token 都重新计算完整前缀。
2. 缓存历史 token 的 `K` 和 `V`，只计算新 token 的 query。

比较两种方式在不同 prompt 长度和生成长度下的 latency。

### 验收标准

- [ ] 两种方式生成结果一致或在随机性控制下等价。
- [ ] 能说明 KV cache 保存的具体 tensor shape。
- [ ] 能解释 cache 降低重复计算但增加显存占用。
- [ ] 能说明 batch、head、sequence length 对 cache 大小的影响。

---

## Lab 23 — `torch.compile` 优化实验

### 开启条件

完成 Lab 18，并能解释一次 forward 中包含哪些主要算子。

### 题干

比较 eager model 与 `torch.compile(model)` 的训练或推理性能。使用 profiler 观察编译前后的 operator/kernel 数量、GPU 时间、CPU launch overhead 和首次运行开销。

### 验收标准

- [ ] 单独记录首次运行和稳定运行阶段。
- [ ] 使用充分 warmup，并说明测量区间。
- [ ] 报告 latency、吞吐量和显存变化。
- [ ] 能解释图捕获、算子融合和编译开销。
- [ ] 能指出 dynamic shape 或控制流可能造成的限制。

---

# Part V：CS336 Scaling、Data 与 Post-training 扩展

这一部分不是进入 CS336 的前置必需内容。完成 Systems 主线后，按照自己的研究方向选择。对于当前 AI Infra 目标，优先级低于 Lab 18–23。

## Lab 24 — Scaling Law 小实验

### 开启条件

理解训练 loss、模型规模、数据量和 compute budget 的基本关系。

### 题干

使用多个很小的模型和训练预算运行实验，记录参数量、训练 token 数、训练时间和最终 loss，拟合一个简化的 scaling trend。重点是设计可比较的实验，而不是得到具有论文精度的结论。

### 验收标准

- [ ] 改变一个变量时尽量控制其他变量。
- [ ] 记录配置、随机种子和训练预算。
- [ ] 绘制模型规模/数据量与 loss 的关系。
- [ ] 能说明小规模实验为什么不能直接外推到大模型。

---

## Lab 25 — 数据过滤、去重与 Packing Pipeline

### 开启条件

开始 CS336 Data 内容，已经具备 Python 文件处理和基础数据统计能力。

### 题干

实现一个小型文本数据处理 pipeline：读取原始文本，进行长度/字符过滤，计算 hash 完成精确去重，再将数据 tokenize 并 pack 成训练样本。记录每一步的数据量变化。

### 验收标准

- [ ] pipeline 可重复运行。
- [ ] 每个处理阶段有输入/输出计数。
- [ ] 去重逻辑不会误删明显不同的样本。
- [ ] 生成的数据能够被 Mini-GPT 读取。
- [ ] 能解释数据质量、数据量和训练吞吐量之间的关系。

---

## Lab 26 — LoRA Linear

### 开启条件

学过低秩分解或参数高效微调；这部分与 CS336 post-training 内容相衔接。

### 题干

实现：

$$
y=x\left(W+\frac{\alpha}{r}BA\right)^T+b
$$

冻结原始 `W` 和 `b`，只训练低秩矩阵 `A、B`。在小型回归任务上比较全参数训练与 LoRA 的可训练参数数量、显存和 loss。

### 验收标准

- [ ] 原始 `W` 不被 optimizer 更新。
- [ ] `A、B` 的 shape 正确。
- [ ] 报告参数量和可训练参数减少比例。
- [ ] LoRA loss 能在小任务上下降。
- [ ] 能解释参数减少不等于计算和显存完全按相同比例减少。

---

## Lab 27 — Toy SFT / Preference Loss

### 开启条件

学过 SFT、chosen/rejected response 或 DPO 的基本概念。

### 题干

不训练完整语言模型，只给定 chosen 和 rejected response 的 token log-probability，实现一个 token-level SFT loss 和一个简化 preference loss。验证 loss 对 chosen/rejected 分数变化的方向符合预期。

### 验收标准

- [ ] 正确处理 padding 或 response mask。
- [ ] chosen 得分提高时 preference loss 不应变差。
- [ ] 能解释 token-level loss 如何聚合为 sequence-level score。
- [ ] 能说明 toy loss 与完整 DPO/SFT 实现的差异。

---

## Lab 28 — Toy RLVR 闭环（可选）

### 开启条件

完成 Lab 27，并已经理解 reward、sampling、policy update 的基本概念。

### 题干

为一个极小的数学字符串任务构造可验证 reward：模型生成答案后，由程序判断答案是否正确，再统计不同策略或训练阶段的准确率变化。第一版只实现采样、奖励统计和离线分析，不直接进行复杂在线 RL。

### 验收标准

- [ ] reward 规则完全可复现。
- [ ] 能区分生成质量、reward hacking 和评估指标。
- [ ] 保存每次采样的输入、输出、reward 和模型版本。
- [ ] 能说明从 toy RL 到真实 RLVR 还缺少哪些组件。

---

# Part VI：研究小项目——LatentScale

本部分是对 `Lab 24` 的研究级升级。项目名称暂定为 `LatentScale`，中文可以叫“模型泛化过程中的表示空间研究”。

这里的核心不是把中间结果画成漂亮的二维图，而是建立一套测量方法，回答一个可以被实验验证的问题：

> **模型规模、数据量和训练计算量增加时，中间表示空间会如何变化？这些变化能否解释普通测试集和分布外测试集上的泛化能力？**

项目最后允许得到“没有找到统一规律”这样的结论，只要实验设计严谨、记录完整、结论与证据匹配。研究项目的价值不在于一定得到漂亮结果，而在于你能否提出可验证、可重复、可以被否定的问题。

## Research Lab 01 — 为模型安装“观察仪器”

### 开启条件

- 完成 `Lab 13`，能够训练一个 Tiny Transformer。
- 能解释模型中 embedding、attention、MLP 和 LM head 的基本作用。
- 能读取一个模型的参数和 forward 输出。

### 题干

为 Tiny Transformer 增加一个表示采集器。它在指定输入上运行模型，并保存以下内容：

- 输入 token 或题目编号。
- 模型预测结果和 loss。
- 每一层 Transformer block 的输出。
- 采集时的训练 step、模型配置和随机种子。

建议把每次采集结果保存为一个独立文件，并使用类似下面的目录：

```text
latent-scale/
├── configs/
├── checkpoints/
├── representations/
│   ├── step_000100/
│   ├── step_001000/
│   └── step_010000/
├── metrics/
└── reports/
```

### 需要先想清楚的问题

同一个输入包含多个 token 时，不能直接把所有 token 的向量混在一起。第一版选择一种固定规则，并在文档中写清楚，例如：

- 只取最后一个 token 的中间表示；或
- 对所有 token 的表示求平均；或
- 只研究某个固定位置的表示。

第一版建议选择“最后一个位置的表示”，因为语言模型最后一个位置通常直接参与下一个 token 的预测，比较容易解释。

### 验收标准

- [ ] 相同模型、相同输入和相同检查点重复采集时，结果一致。
- [ ] 能明确写出保存数组的 shape，例如 `[样本数, 隐藏维度]`。
- [ ] 能说明每一层保存的是 block 前还是 block 后的结果。
- [ ] 采集过程不会改变模型参数，也不会产生梯度。
- [ ] 能从保存文件中恢复某一个样本的中间表示。
- [ ] 保存配置、step、模型大小和数据集版本，保证以后知道这份结果来自哪里。

### 交付物

1. `RepresentationProbe` 或同等功能的代码。
2. 一份小型表示文件。
3. 一页说明文档，解释采集位置、shape 和保存格式。

---

## Research Lab 02 — 先研究“记忆”和“学会规律”的区别

### 开启条件

完成 `Research Lab 01`，并完成 `Lab 16` 的基本检查点保存功能。

### 题干

构造一个很小、规律明确的人工任务。推荐使用模运算：

$$
(a+b)\bmod p=c
$$

例如，输入是一组数字 `a、b`，目标是预测结果 `c`。训练集只包含部分组合，测试集包含训练时没有出现过的组合。

需要明确区分三类数据：

1. **训练集**：模型真正用来更新参数的数据。
2. **普通测试集**：格式与训练集相同，但组合没有出现过。
3. **变形测试集**：在输入范围、组合方式或排列形式上做有计划的变化，用来测试更强的泛化能力。

### 必须记录的曲线

每隔固定 step 记录：

- 训练集准确率。
- 普通测试集准确率。
- 变形测试集准确率。
- 训练 loss 和测试 loss。
- 每一层表示的基本统计量。

你可能观察到这样的过程：模型先把训练题做对，但测试题仍然做错；继续训练后，测试准确率突然明显提升。这种现象常被称为 grokking，本文暂时只把它作为“泛化阶段突然出现”的直观例子，不要求现在专门研究它。

### 验收标准

- [ ] 能准确列出训练集和每个测试集分别包含哪些样本。
- [ ] 能证明测试集没有误混入训练集。
- [ ] 固定随机种子后，实验可以重复运行。
- [ ] 至少保存 5 个不同训练阶段的检查点和评估结果。
- [ ] 能画出训练准确率、普通测试准确率和变形测试准确率随 step 的变化。
- [ ] 能指出模型何时可能从“记忆”进入“学规律”阶段。

### 重要提醒

不要因为一张二维图出现了几个簇，就直接说模型“理解了规律”。二维降维方法会主动改变点之间的关系，只能用于辅助观察，不能作为主要证据。项目的主要结论必须来自可计算的指标和对照实验。

---

## Research Lab 03 — 测量表示空间的基本结构

### 开启条件

完成 `Research Lab 02`，会使用 NumPy 或 PyTorch 对 `[样本数, 隐藏维度]` 的矩阵进行统计。

### 题干

对每个训练阶段、每一层和每一种数据集，计算下面三个第一版指标。

#### 1. 有效维度

设所有样本的表示组成矩阵 `Z`。先减去每一列的平均值，得到中心化矩阵，再计算协方差矩阵的特征值：

$$
\lambda_1,\lambda_2,\ldots,\lambda_d
$$

用这些特征值估计有效维度：

$$
R_{\mathrm{eff}}=
\frac{\left(\sum_i\lambda_i\right)^2}
{\sum_i\lambda_i^2}
$$

它可以粗略表示：数据的变化主要集中在多少个方向上。数值变小不一定代表模型更好，只能说明表示的变化更集中。

#### 2. 邻居保持率

对每个样本，找到表示空间中距离最近的 `k` 个样本，记录这些邻居是否具有相同的规律类别、相同的结果或相同的结构属性。

例如，如果两个输入只改变了数字顺序，但它们在表示空间中仍然靠近，说明模型可能保留了某种相似关系。

#### 3. 训练点与新点的距离

对每个测试样本，计算它与训练表示集合中最近点的距离：

$$
d_{\mathrm{train}}(z)=\min_{z_i\in Z_{\mathrm{train}}}\|z-z_i\|_2
$$

然后比较这个距离与模型是否预测正确之间的关系。

### 第一版只做什么

第一版只做上面三个指标，不要一开始同时加入十几个复杂指标。每个指标都必须回答：

- 它的输入是什么？
- 它的计算公式是什么？
- 它的数值变大或变小表示什么？
- 它有没有受到隐藏维度、样本数量和距离尺度的影响？

### 验收标准

- [ ] 每个指标都能在一个人工构造的小矩阵上验证。
- [ ] 指标计算不会使用测试标签参与模型训练。
- [ ] 指标结果按 `step × layer × dataset` 保存。
- [ ] 能比较训练集与普通测试集的距离分布。
- [ ] 能说明有效维度低不等于一定泛化好。
- [ ] 报告中同时给出均值、标准差和样本数量，避免只展示一个数字。

### 可选扩展：局部维度

完成第一版后，可以研究每个样本附近的局部维度，而不是对整个数据集只算一个总数。这个方向更接近当前研究文献，但实现和解释都更容易出错，因此必须放在基础指标之后。

---

## Research Lab 04 — 观察训练过程中空间结构何时改变

### 开启条件

完成 `Research Lab 03`，并能从 `Lab 16` 的检查点恢复模型。

### 题干

对多个训练阶段重复执行表示采集和指标计算，形成如下实验表：

| 训练 step | 层 | 数据集 | 训练准确率 | 测试准确率 | 有效维度 | 邻居保持率 | 最近训练点距离 |
|---:|---:|---|---:|---:|---:|---:|---:|
| 100 | 0 | train |  |  |  |  |  |
| 100 | 0 | test |  |  |  |  |  |
| 1000 | 0 | train |  |  |  |  |  |
| 1000 | 0 | test |  |  |  |  |  |

至少观察：

- 早期训练阶段。
- 训练集已经较好但测试集仍较差的阶段。
- 测试集开始提升的阶段。
- 训练结束阶段。

### 要回答的问题

1. 哪一层最早出现可观察的空间结构变化？
2. 空间指标的变化是否早于测试准确率提升？
3. 训练集和测试集的距离分布是否逐渐变得相似？
4. 不同指标是否给出一致结论？

### 验收标准

- [ ] 至少有 5 个训练阶段、至少 3 层表示和 3 类数据集。
- [ ] 所有曲线使用相同的横轴定义和相同的评估数据。
- [ ] 明确区分“先发生”与“相关”：时间上先发生不代表一定造成后续结果。
- [ ] 至少找到一个指标与泛化变化不同步的反例，并解释它为什么不能单独作为判断标准。
- [ ] 生成一份包含实验设置、曲线、观察和局限性的短报告。

---

## Research Lab 05 — 模型规模、数据量和训练计算量对比

### 开启条件

完成 `Lab 24` 和 `Research Lab 04`。

### 题干

设计一个小规模对照实验，分别改变：

- 模型大小，例如隐藏维度或层数。
- 训练数据量。
- 训练步数或总 token 数。

每次实验记录：

- 参数量。
- 训练样本或 token 数。
- 训练时间和 GPU 时间（如果有 GPU）。
- 训练集、普通测试集和变形测试集的结果。
- 表示空间指标。

推荐先做一个很小的网格，而不是一次运行几十组：

| 实验组 | 模型大小 | 数据量 | 训练预算 |
|---|---|---:|---:|
| A | 小 | 小 | 固定 |
| B | 大 | 小 | 固定 |
| C | 小 | 大 | 固定 |
| D | 大 | 大 | 固定 |

然后再增加“固定总计算量”的比较，研究在预算相近时，扩大模型和增加数据哪一种方式更容易得到好的泛化。

### 注意控制变量

如果模型变大、数据变多、训练时间也同时增加，最后即使结果变好，你也无法知道是哪一个因素起主要作用。因此每轮实验必须写出：

- 哪个变量被改变。
- 哪些变量被固定。
- 这个实验要排除哪一种解释。

### 验收标准

- [ ] 至少完成 4 组可比较实验。
- [ ] 每组实验都有配置文件、随机种子和检查点说明。
- [ ] 同时报告 loss、普通测试结果、变形测试结果和表示指标。
- [ ] 至少比较一次“相近测试 loss、但变形测试结果不同”的模型。
- [ ] 不用单条曲线就宣称存在 scaling law，能说明样本规模和实验范围的局限。
- [ ] 能回答：模型规模和数据量是否通过相同的空间结构变化改善泛化？

---

## Research Lab 06 — 把研究变成一个可复现的测量系统

### 开启条件

完成 `Research Lab 01–05`，并完成 `Lab 17–18` 中至少一个 profiling 实验。

### 题干

把前面零散的脚本整理成一个小型实验系统，至少支持：

```text
读取配置
    ↓
训练模型
    ↓
保存检查点
    ↓
采集中间表示
    ↓
计算空间指标
    ↓
生成表格和图
    ↓
输出实验报告
```

每次实验都应当能够通过一个配置文件复现，配置中至少包含：

- 模型结构。
- 数据集版本和划分方式。
- 训练步数、batch size 和学习率。
- 随机种子。
- 表示采集的层和训练 step。
- 指标计算参数，例如邻居数 `k`。

### Infra 角度的测量

研究系统也要测量自身成本：

- 采集中间表示需要多长时间。
- 表示文件占用多少磁盘空间。
- 采集是否增加 GPU 显存。
- 指标计算在 CPU 或 GPU 上分别需要多长时间。
- 表示采集频率改变后，训练总时间增加多少。

可以定义一个简单的效率指标：

$$
\text{单位计算成本获得的泛化提升}
=
\frac{\text{测试指标提升}}
{\text{额外训练或分析时间}}
$$

这个指标不是标准学术定义，重点是训练你从“结论是否正确”进一步思考“得到结论需要付出多少系统成本”。

### 验收标准

- [ ] 新增一个配置文件即可运行一组新实验。
- [ ] 实验结果包含配置、日志、检查点、指标和图表之间的对应关系。
- [ ] 删除中间缓存后仍能从检查点重新生成分析结果。
- [ ] 对表示采集和指标计算做一次 profiling。
- [ ] 能指出系统中最耗时、最占显存和最占磁盘的环节。
- [ ] 写出最终报告，并明确区分实验事实、推测和不能得出的结论。

## 这个小项目的最终报告结构

完成 `Research Lab 01–06` 后，报告建议使用下面的结构：

1. **问题**：模型泛化时表示空间是否发生系统变化？
2. **实验任务**：为什么使用人工任务，训练集和测试集如何划分？
3. **模型与训练**：模型结构、数据量、训练预算和随机种子。
4. **表示采集方法**：采集哪一层、哪个位置、保存什么 shape。
5. **指标定义**：有效维度、邻居保持率和最近训练点距离的公式与解释。
6. **训练动态**：空间指标与训练/测试结果随 step 如何变化？
7. **规模对照**：模型大小、数据量和计算量分别造成了什么变化？
8. **反例与局限**：哪些情况下指标没有解释泛化？实验还缺少什么？
9. **系统成本**：采集、存储和分析的时间及资源开销。
10. **结论**：只写实验能够支持的结论，不把相关性写成因果性。

## 研究项目的最低完成线、标准完成线和扩展方向

### 最低完成线

完成 `Research Lab 01–03`，能够采集中间表示并计算至少两个空间指标。

### 标准完成线

完成 `Research Lab 01–06`，能够展示训练动态、规模对照、资源开销和可复现实验流程。

### 扩展方向

在标准完成线之后，才考虑以下扩展：

- 比较不同模型结构，而不是只改变宽度和层数。
- 加入更复杂的组合任务或小型文本任务。
- 研究表示相似性：比较不同训练阶段或不同模型的表示是否相近。
- 将同样的表示测量方法用于教师模型和学生模型，研究蒸馏是否保留了教师的泛化结构。
- 将空间指标加入训练监控，测试它能否帮助选择检查点或决定是否继续训练。

其中最后两项可以与后续的模型蒸馏项目连接起来：普通蒸馏通常只比较模型输出是否接近，而这里进一步询问学生模型是否保留了教师模型中与泛化有关的空间结构。这个方向必须在基础项目完成后再做，第一版不加入复杂的特征蒸馏模块。

---

# 7. 最终路线与优先级

## 当前必须完成

```text
Assignment 2
→ Lab 01
→ Lab 02
→ Lab 03
```

## CS224N 基础主线

```text
Lab 04 Embedding
→ Lab 06 RNN Forward
→ Lab 07 BPTT
→ Lab 08 Toy Attention
→ Lab 09 C++ Causal Attention
→ Lab 11 Multi-Head Layout
→ Lab 12 Tensor Layout
→ Lab 13 Tiny Transformer
```

Lab 05 是较独立的 C++ 小题，可以在词向量内容之后穿插完成；Lab 10 是 Lab 09 的系统优化扩展。

## 进入 CS336 的核心主线

```text
Lab 14 Mini-GPT 独立重建
→ Lab 15 Tokenizer / Packing
→ Lab 16 可靠训练循环
→ Lab 17 Resource Accounting
→ Lab 18 Benchmark / Profiler
→ Lab 19 Triton
→ Lab 20 Memory-efficient Attention
→ Lab 21 DDP
→ Lab 22 KV Cache
→ Lab 23 torch.compile
```

## 可选扩展主线

```text
Lab 24 Scaling Law
→ Lab 25 Data Pipeline
→ Lab 26 LoRA
→ Lab 27 SFT / Preference Loss
→ Lab 28 Toy RLVR
```

## 研究小项目的解锁顺序

```text
Lab 13 Tiny Transformer
    ↓
Research Lab 01 表示采集
    ↓
Lab 16 可靠训练循环
    ↓
Research Lab 02 记忆与泛化
    ↓
Research Lab 03–04 空间指标与训练动态
    ↓
Lab 17–18 资源统计与 profiling
    ↓
Lab 24 Scaling Law
    ↓
Research Lab 05–06 规模对照与可复现系统
```

这条支线不改变原来的学习顺序。当前仍然只执行：

```text
Assignment 2
→ Lab 01
→ Lab 02
```

## 7. 完成这份路线后的能力目标

完成 CS224N 基础主线后，应当能够：

- 独立解释反向传播、RNN、Attention 和 Transformer 的核心数据流。
- 用 C++ flat memory 写出小型算子。
- 用 NumPy 验证数学公式和 shape。
- 用 PyTorch 搭建并训练 Tiny Transformer。

完成 CS336 核心主线后，应当能够：

- 独立组织一个小型语言模型训练项目。
- 正确实现 checkpoint、gradient accumulation、mixed precision 和基本评估。
- 使用 benchmark 和 profiler 定位性能瓶颈。
- 理解 kernel、显存访问、算子融合、Triton、DDP 和 KV cache 的基本关系。
- 从“模型代码”进一步追踪到“算子、kernel 和硬件”。

完成研究小项目的标准版本后，还应当能够：

- 从模型检查点中采集并解释中间表示。
- 使用可计算指标描述表示空间，而不是只依赖二维可视化。
- 设计训练集、普通测试集和变形测试集，区分记忆与泛化。
- 研究模型规模、数据量和训练预算对泛化的影响。
- 用配置、日志、检查点和 profiling 结果复现实验。
- 明确区分实验事实、统计相关性、研究假设和暂时不能回答的问题。

这时再进入超分模型蒸馏、推理优化、端侧部署或后训练项目，学习重点就可以从“补基础”逐渐转向真实 AI Infra 工程问题。
