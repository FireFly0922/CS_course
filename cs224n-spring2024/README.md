# Stanford CS224N — NLP with Deep Learning（Spring 2024）

Stanford **CS224N: Natural Language Processing with Deep Learning**，Spring 2024（term 1246）。
主讲：**Chris Manning**、**Shikhar Murty**。官网存档：<https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/>

本目录包含：**Slides 幻灯片**、**Readings 阅读资料（课程笔记 + 论文）**、**Assignments 作业**、**Final Project 项目文档**、辅助材料。

---

## 目录结构

```
cs224n-spring2024/
├── slides/          18 讲幻灯片 PDF (cs224n-spr2024-lectureNN-*.pdf)
├── readings/
│   ├── notes/       11 个课程笔记(词向量/神经网络/依存句法/RNN/自注意力/Python复习/数学复习)
│   └── papers/      40 篇指定阅读论文(word2vec/GloVe/Transformer/BERT 等)
├── assignments/     a1–a4(notebook/handout/code/LaTeX 模板)
├── project/         期末项目 5 个文档(handout/proposal/milestone/report/tips)
├── materials/       Gensim 词向量可视化示例
└── README.md
```

---

## 课程安排 · 18 讲

| # | 日期 | 主题 | 幻灯片 |
|---|------|------|--------|
| 1 | 4/2 | Word Vectors 词向量 | `lecture01-wordvecs1` |
| 2 | 4/4 | Word Vectors and Language Models | `lecture02-wordvecs2` |
| 3 | 4/9 | Backpropagation and Neural Network Basics | `lecture03-neuralnets` |
| 4 | 4/11 | Dependency Parsing 依存句法分析 | `lecture04-dep-parsing` |
| 5 | 4/16 | Recurrent Neural Networks | `lecture05-rnnlm` |
| 6 | 4/18 | Seq2Seq 与机器翻译 | `lecture06-fancy-rnn` |
| 7 | 4/23 | Final Projects and LLM intro | `lecture07-final-project` |
| 8 | 4/25 | Transformers（Anna Goldie）| `lecture08-transformers` |
| 9 | 4/30 | Pretraining 预训练 | `lecture09-pretraining` |
| 10 | 5/2 | Post-training（RLHF/SFT/DPO，Archit Sharma）| `lecture10-prompting-rlhf` |
| 11 | 5/7 | Benchmarking and Evaluation（Yann Dubois）| `lecture11-evaluation-yann` |
| 12 | 5/9 | Efficient NN Training（Shikhar Murty）| `lecture12-training-shikhar` |
| 13 | 5/14 | Speech Brain-Computer Interface（Chaofei Fan）| `lecture13-speech-bci` |
| 14 | 5/16 | Reasoning and Agents（Shikhar Murty）| `lecture14-agents-shikhar` |
| 15 | 5/21 | Life after DPO（Nathan Lambert）| `lecture15-life-after-dpo-lambert` |
| 16 | 5/23 | ConvNets, Tree-RNN 与成分句法 | `lecture16-CNN-TreeRNN` |
| 17 | 5/28 | Responsible NLP（Adina Williams）| `lecture17-human-centered-nlp` |
| 18 | 5/30 | NLP, Linguistics, and Philosophy | `lecture18-nlp-linguistics-philosophy` |

---

## 作业 Assignments（a1–a4 + 期末项目）

| 作业 | 内容 | 文件 |
|------|------|------|
| a1 | 词向量探索（Jupyter notebook）| `a1.zip`、`a1_preview_exploring_word_vectors.html` |
| a2 | word2vec 数学推导 + 实现 | `a2.pdf`(handout)、`a2.zip`(code)、`a2_latex_template.zip` |
| a3 | 依存句法分析器（PyTorch）| `a3_spr24_student_handout.pdf`、`_code.zip`、`_latex.zip` |
| a4 | 神经机器翻译（seq2seq + attention）| `a4_spr24_student_handout.pdf`、`_code.zip` |
| 项目 | 默认项目 minBERT / 自定义项目 | `project/` 下 handout、proposal、milestone、report instructions、tips |

---

## 阅读资料 Readings

- **课程笔记 `readings/notes/`**：词向量、神经网络与反向传播、依存句法、语言模型与 RNN、自注意力与 Transformer（草稿）、Python 复习、梯度/微积分复习等。
- **指定论文 `readings/papers/`**（40 篇）：word2vec（Mikolov）、GloVe（Pennington）、Transformer《Attention is All You Need》、BERT、Seq2Seq、Attention（Bahdanau）、Layer Norm、Jurafsky SLP 教材章节、fairmlbook 公平性等，覆盖每讲的 Suggested Readings。
  > 少数论文为外部链接（arxiv/ACL/各作者主页）；arxiv 链接已规范化为 `/pdf/` 直下。

---

*资料来源：web.stanford.edu/class/archive/cs/cs224n/cs224n.1246（Stanford CS224N Spring 2024）。仅供个人学习，版权归 Stanford 及各作者所有。*
