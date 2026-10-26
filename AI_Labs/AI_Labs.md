# LLM 基础与推理加速执行计划

## 1. 最终目标

建立完整的 LLM 训练与推理知识链，并完成一个可用于推理加速 / AI Infra 实习申请的 MiniServe 项目。

```text
数学与模型结构
    → Tensor shape / stride / dtype
    → PyTorch operator
    → 训练与生成流程
    → FLOPs / Memory / Bandwidth
    → Kernel / Compiler
    → Inference Runtime / Serving
```

## 2. 教学与实践方式

每个知识点按照以下顺序学习：

1. 中文解释与英文术语对照；
2. 数学公式和 tensor shape；
3. 小规模手算或 NumPy/PyTorch 实验；
4. 学习者完成首版实现；
5. 单元测试、边界测试与 code review；
6. 复杂度、内存访问和性能分析；
7. 整理为后续项目的可复用模块。

AI 辅导内容包括题目翻译、概念讲解、API 说明、测试设计、错误定位、代码审阅和性能分析。CS336 正式作业的核心实现由学习者完成。

## 3. 课程资料映射

| 阶段 | Datawhale Diy-LLM | CS336 | 主要目标 |
|---|---|---|---|
| Phase 0 | 第 3 章 PyTorch 与资源核算 | 前置能力 | 熟悉 PyTorch 训练原语 |
| Phase 1 | 第 4 章模型架构 | Assignment 1 前置 | 理解 Transformer forward |
| Phase 2 | 第 2–4 章 | Assignment 1 Basics | 训练 TinyLM |
| Phase 3 | 第 3、6、7 章 | Assignment 2 前半 | 建立性能分析方法 |
| Phase 4 | 第 10 章推理 | 推理专项 | 掌握 KV Cache 与推理指标 |
| Phase 5 | 第 6–8 章 | Assignment 2 Systems | 学习 kernel、编译与并行 |
| Phase 6 | 第 10、12 章 | 综合项目 | 构建 MiniServe |

## 4. Phase 0 — PyTorch Training Bridge

### 目标

掌握 tensor、autograd、`nn.Module`、训练循环、评估循环和基础资源核算。

### 项目路径

`CS336_projects/phase0-pytorch-bridge/`

### 任务

1. 构造一个二维二分类数据集。
2. 实现以下模型：

```text
[B, 2]
  → Linear(2, 16)
  → ReLU
  → Dropout
  → Linear(16, 2)
  → logits [B, 2]
```

3. 完成训练循环：

```text
zero_grad
→ forward
→ cross entropy
→ backward
→ optimizer.step
```

4. 完成 `eval()` 与 `no_grad()` 评估。
5. 打印核心 tensor 的 shape、dtype、device 和 stride。
6. 检查参数梯度的 shape 与范数。
7. 计算参数量、FP32 权重字节数与 AdamW 状态开销。
8. 实验 `view`、`reshape`、`transpose`、`contiguous`。

### 验收标准

- loss 明显下降；
- 测试结果明显高于随机猜测；
- 能口述训练循环中数据与梯度的变化；
- 能解释 logits 与 cross entropy 的关系；
- 能解释梯度累积与 `zero_grad`；
- 能解释 `train()`、`eval()` 和 `no_grad()`；
- 能解释 shape 与 stride 的区别；
- 至少包含 3 个自动测试；
- README 记录运行方式、shape、参数量和实验结果。

## 5. Phase 1 — Transformer Foundations

### 目标

理解 token 进入 Transformer 后的完整 forward 数据流。

### 学习任务

1. Embedding lookup 与 token id。
2. Scaled dot-product attention：

$$
S=\frac{QK^T}{\sqrt{d_k}},\qquad
P=\operatorname{softmax}(S),\qquad
Y=PV
$$

3. Causal mask 与数值稳定 softmax。
4. Multi-head attention 的 reshape、transpose 和拼接。
5. RMSNorm、SwiGLU/MLP、残差连接。
6. `[B,T] → [B,T,C] → [B,T,V]` 数据流。
7. next-token prediction 与 target shift。

### 验收项目

`CS336_projects/transformer-foundations/`

项目包含：

- NumPy 单头 Attention；
- PyTorch causal multi-head attention；
- tensor layout 实验；
- 一个可以过拟合极小数据集的 Tiny Transformer。

### 验收标准

- attention 每行概率和接近 1；
- causal mask 后未来位置概率接近 0；
- 每一步 reshape/transpose 的 shape 和 stride 清晰；
- Tiny Transformer 的训练 loss 明显下降；
- 能从 token id 解释到 logits。

## 6. Phase 2 — TinyLM / Diy-LLM Basics

### 目标

完成一个可训练、可保存、可恢复、可生成文本的小型语言模型。

### 项目路径

`CS336_projects/tinylm/`

### 模块

```text
Tokenizer
→ Dataset Packing
→ Transformer LM
→ Cross Entropy
→ AdamW / Scheduler
→ Training Loop
→ Checkpoint
→ Generation / Evaluation
```

### 验收标准

- tokenizer 可以 encode/decode；
- input 与 target 正确错开一个 token；
- 模型通过选定的 CS336 Assignment 1 测试；
- checkpoint 可以恢复模型、优化器和训练 step；
- 固定数据上的 loss 明显下降；
- 可以生成短文本；
- README 解释模型配置、参数量、训练配置和结果。

## 7. Phase 3 — Performance Foundations

### 目标

建立测量、分析与验证的性能工程方法。

### 项目路径

`CS336_projects/performance-foundations/`

### 学习任务

1. 估算 Linear、MLP、Attention 的参数量、激活量和 FLOPs。
2. 估算权重、激活和中间结果的字节数。
3. 计算算术强度：

$$
AI=\frac{\text{FLOPs}}{\text{Bytes moved}}
$$

4. 正确使用 warmup、GPU synchronize 和重复测量。
5. 区分 CPU wall time、GPU kernel time 与端到端时间。
6. 使用 PyTorch Profiler 分析训练和推理步骤。

### 验收标准

- 至少比较两个 batch size、两个 sequence length 和两个 dtype；
- 保存原始 CSV/JSON 数据；
- 报告 latency、throughput 和峰值显存；
- profiler 报告指出主要算子和瓶颈；
- 理论估算与实测差异具有解释。

## 8. Phase 4 — Inference Core

### 目标

掌握自回归推理的核心数据流和性能瓶颈。

### 项目路径

`CS336_projects/inference-core/`

### 实现

1. Naive generation：每步重新计算完整前缀。
2. KV Cache generation：复用历史 token 的 K/V。
3. Prefill 与 Decode 分离计时。
4. 计算每层 KV Cache shape 与总字节数。
5. 测量 TTFT、TPOT、tokens/s 和峰值显存。

### 对照变量

- batch size；
- prompt length；
- generation length；
- dtype；
- attention head 数量与 head dimension。

### 验收标准

- Naive 与 KV Cache 在确定性解码下输出一致；
- cache 随序列增长的 shape 正确；
- 理论 cache 大小与实测结果可以对照；
- 报告 Prefill 与 Decode 的性能差异；
- 能解释 KV Cache 的计算收益与内存/带宽成本。

## 9. Phase 5 — Systems Performance

### 目标

从 PyTorch operator 深入到 kernel、内存访问和编译优化。

### 项目路径

`CS336_projects/systems-performance/`

### 学习任务

1. Triton vector add 或 elementwise kernel。
2. Triton row-wise softmax。
3. 分块、数值稳定的 memory-efficient attention。
4. PyTorch eager 与 `torch.compile` 对照。
5. 根据服务器条件完成 DDP/FSDP 基础实验。

### 验收标准

- 自定义实现与 PyTorch 基线数值一致；
- benchmark 覆盖多个 shape；
- profiler 展示 kernel 数量、GPU 时间和 launch overhead；
- 能解释 block、mask、pointer arithmetic、tiling 和 fusion；
- 性能报告包含优化有效区间与边界。

## 10. Phase 6 — MiniServe

### 目标

交付一个可展示、可复现、可讲解的小型 LLM 推理 runtime。

### 项目结构

```text
CS336_projects/miniserve/
├── README.md
├── pyproject.toml
├── src/
│   ├── model/
│   ├── engine/
│   └── kernels/
├── tests/
├── benchmarks/
├── configs/
├── profiles/
├── results/
└── docs/
```

### 核心功能

- 模型 forward 与 generation；
- Prefill / Decode；
- KV Cache 管理；
- 多请求 batching；
- 显存预算；
- Triton 或编译优化；
- 可复现 benchmark。

### 扩展方向

- Paged KV Cache；
- 权重量化或 KV Cache 量化；
- continuous batching；
- prefix caching；
- speculative decoding；
- tensor parallelism。

### 求职验收标准

1. 陌生读者可以通过 README 完成安装和运行。
2. 测试覆盖正确性、边界条件和基线对照。
3. 性能表包含 GPU、CUDA、PyTorch、dtype、shape、warmup 和重复次数。
4. 报告 TTFT、TPOT、tokens/s、峰值显存和 KV Cache 开销。
5. profiler 证据能够解释主要瓶颈与优化收益。
6. 架构文档说明模块边界和关键数据流。
7. 优化日志记录假设、实验、结果与结论。
8. 简历描述使用真实实验数据。

简历描述结构：

> 实现的系统 + 目标瓶颈 + 优化方法 + 硬件与输入配置 + 指标改善 + 正确性保证。

## 11. 环境分工

### 本地

- Python / PyTorch 学习；
- CPU 正确性测试；
- 小 shape 调试；
- 单元测试；
- 文档与 Git 管理。

### 服务器

- CUDA 与 Triton；
- PyTorch Profiler / Nsight；
- 大 shape benchmark；
- 多 GPU 实验；
- MiniServe 最终性能数据。

服务器实验统一记录：

- GPU 型号；
- 驱动、CUDA、PyTorch、Triton 版本；
- dtype 与输入 shape；
- warmup 与正式重复次数；
- 首次运行和稳定态；
- 原始数据文件与统计方法。

## 12. 进度表

| 阶段 | 状态 | 完成日期 |
|---|---|---|
| Phase 0 — PyTorch Training Bridge | 🔓 | — |
| Phase 1 — Transformer Foundations | ⬜ | — |
| Phase 2 — TinyLM / Diy-LLM Basics | ⬜ | — |
| Phase 3 — Performance Foundations | ⬜ | — |
| Phase 4 — Inference Core | ⬜ | — |
| Phase 5 — Systems Performance | ⬜ | — |
| Phase 6 — MiniServe | ⬜ | — |

## 13. 当前任务

当前执行 Phase 0：

```text
Tensor 基础
→ 两层网络 forward
→ CrossEntropyLoss
→ backward 与 AdamW
→ evaluation
→ shape / stride 实验
→ 参数量与显存核算
→ 自动测试与 README
```
