# CS336: Language Modeling from Scratch — Spring 2026

Stanford CS336「从头构建大语言模型」2026 春季版全套资料存档。
讲师：**Tatsunori Hashimoto**、**Percy Liang**。官网：<https://stanford-cs336.github.io/>
课程录像：官方 YouTube playlist（见官网 Recordings）。

本目录包含：**Lectures 讲义（PDF slides + 可运行 Python 讲义）**、**5 次 Assignments 作业（handout + starter code）**、**Readings 阅读论文（98 篇 PDF + 13 个网页链接）**。

---

## 目录结构

```
cs336-2026/
├── lectures/          讲义仓库(stanford-cs336/lectures)完整内容
│   ├── lecture_03/04/05/08/09/11/15/16.pdf   幻灯片(8讲)
│   ├── lecture_01/02/06/07/10/12/13/14/17.py 可运行讲义(9讲, 见下)
│   ├── references.py  全部阅读引用(111条)的结构化清单
│   └── (facts.py, lecture_util.py, assets/, images/, pyproject.toml…)
├── assignments/       5 次作业(各含 handout PDF + starter code)
│   ├── assignment1-basics/     cs336_assignment1_basics.pdf
│   ├── assignment2-systems/    cs336_assignment2_systems.pdf
│   ├── assignment3-scaling/    cs336_assignment3_scaling.pdf
│   ├── assignment4-data/       cs336_assignment4_data.pdf
│   └── assignment5-alignment/  cs336_spring2026_assignment5_alignment.pdf (+安全RLHF补充)
└── readings/
    └── papers/        讲义引用的 98 篇论文 PDF
```

> **可运行讲义（.py）说明**：CS336 部分讲次（Percy 讲的）不是幻灯片，而是"可执行讲义"——
> Python 源码运行后生成带代码/输出的交互式 trace。可直接当源码阅读；若要运行，用讲义仓库里的
> `uv`（`pyproject.toml`/`uv.lock`）环境。官网在线渲染版：`stanford-cs336.github.io/lectures/?trace=lecture_01`。

---

## 课程安排 · 逐讲（17 讲正课 + 2 场客座）

| # | 日期 | 主题 | 主讲 | 材料 |
|---|------|------|------|------|
| 1 | 3/30 | Overview, tokenization | Percy | `lecture_01.py` · 作业1 出 |
| 2 | 4/1 | PyTorch (einops), resource accounting (FLOPs/内存/算术强度) | Percy | `lecture_02.py` |
| 3 | 4/6 | Architectures, hyperparameters | Tatsu | `lecture_03.pdf` |
| 4 | 4/8 | Attention alternatives 与 mixture of experts | Tatsu | `lecture_04.pdf` |
| 5 | 4/13 | GPUs, TPUs | Tatsu | `lecture_05.pdf` |
| 6 | 4/15 | Kernels, Triton | Percy | `lecture_06.py` · 作业1 截止 / 作业2 出 |
| 7 | 4/20 | Parallelism | Percy | `lecture_07.py` |
| 8 | 4/22 | Parallelism | Tatsu | `lecture_08.pdf` |
| 9 | 4/27 | Scaling laws | Tatsu | `lecture_09.pdf` |
| 10 | 4/29 | Inference | Percy | `lecture_10.py` · 作业2 截止 / 作业3 出 |
| 11 | 5/4 | Scaling laws | Tatsu | `lecture_11.pdf` |
| 12 | 5/6 | Evaluation | Percy | `lecture_12.py` · 作业3 截止 / 作业4 出 |
| 13 | 5/11 | Data (sources, datasets) | Percy | `lecture_13.py` |
| 14 | 5/13 | Data (filtering/dedup/mixing/synthetic) | Percy | `lecture_14.py` |
| 15 | 5/18 | Mid/post-training (SFT/RLHF) | Tatsu | `lecture_15.pdf` |
| 16 | 5/20 | Post-training — RLVR | Tatsu | `lecture_16.pdf` · 作业4 截止 / 作业5 出 |
| 17 | 5/27 | Alignment — multimodality | Percy | `lecture_17.py` |
| 18 | 6/1 | 客座：Daniel Selsam | — | （无讲义） |
| 19 | 6/3 | 客座：Dan Fu | — | 作业5 截止 |

---

## 作业 Assignments（5 次）

| 作业 | 主题 | handout |
|------|------|---------|
| 1 · basics | 从零实现 BPE tokenizer、Transformer、训练循环 | cs336_assignment1_basics.pdf (47页) |
| 2 · systems | 系统优化：Triton kernel、Flash Attention、分布式训练 | cs336_assignment2_systems.pdf (48页) |
| 3 · scaling | Scaling laws 实验与拟合 | cs336_assignment3_scaling.pdf (8页) |
| 4 · data | 数据处理：过滤、去重、质量分类 | cs336_assignment4_data.pdf (16页) |
| 5 · alignment | 对齐：SFT / RLHF / RLVR（含安全 RLHF 补充） | cs336_spring2026_assignment5_alignment.pdf |

每个作业目录 = 官方 GitHub 仓库完整内容（starter code + 测试 + handout）。作业 5 另含测试用 tiny-gpt2 模型与 HH-RLHF 等数据集。

---

## 阅读资料 Readings

`readings/papers/` 下是讲义 `references.py` 引用的 **97 篇论文 PDF**（文件名=引用键，如 `shannon_1950.pdf`、`gpt2_2019.pdf`、`transformer_2017.pdf`）。涵盖：n-gram/神经语言模型经典、Transformer 与各类架构、优化与初始化、GPU/kernel/并行、scaling laws、数据、对齐/RLHF、以及 2024–2026 的 Llama/Qwen/DeepSeek/OLMo 等前沿模型报告。

> 说明：`references.py` 共引用 111 条，其中 98 条为 PDF；`deepseek_v4_2026` 的源链接（HuggingFace `DeepSeek-V4-Pro` 仓库内的 `DeepSeek_V4.pdf`）已失效——该仓库只有模型权重无此 PDF，故未获取（97/98）。`lstm_1997` 原源 `bioinf.jku.at` 无响应，改用 CMU 课程镜像获取。

完整引用清单（含标题/作者/日期）见 `lectures/references.py`。

### 仅链接（网页/博客，未下载 PDF）

- OpenWebText — <https://skylion007.github.io/OpenWebTextCorpus/>
- GPT-J — <https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/>
- The FLOPs Calculus of LM Training — <https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4>
- Alpaca — <https://crfm.stanford.edu/2023/03/13/alpaca.html>
- Transformer Math — <https://blog.eleuther.ai/transformer-math/>
- Muon optimizer — <https://kellerjordan.github.io/posts/muon/>
- Llama 4 — <https://ai.meta.com/blog/llama-4-multimodal-intelligence/>
- OLMo 2 32B — <https://allenai.org/blog/olmo2-32B>
- Marin 8B / 32B — <https://marin.readthedocs.io/>
- Qwen 3.5 — <https://qwen.ai/blog?id=qwen3.5>
- MiniMax M2.5 — <https://www.minimax.io/news/minimax-m25>
- Xiaomi MiMo v2 — <https://mimo.xiaomi.com/mimo-v2-pro>

---

*资料来源：stanford-cs336.github.io 及 github.com/stanford-cs336（Spring 2026）。仅供个人学习，版权归 Stanford 及各作者所有。*
