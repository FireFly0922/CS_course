# LLM 与推理框架学习路线

四个阶段依次推进，近期不并行安排 CS106L 或 LeetCode。具体练习、环境和验收细则放在各阶段对应目录。

## 1. MiniLM：手写并训练语言模型

参考 Diy-LLM、CS336、MiniMind 和 nanoGPT，从数学、tensor shape 和数据流出发，亲手实现 Transformer 核心结构，提升 PyTorch 编码与调试能力。

完成数据处理、Tokenizer、训练循环、从随机初始化开始的预训练和 SFT；根据模型能力与任务需要，选择 DPO、PPO/RLHF 或 GRPO/RLVR 等后训练实验。最终得到能进行基本中文对话、遵循简单指令，并在有限任务上表现出可测能力的小语言模型。根据训练需要学习混合精度、DDP、FSDP 和 DeepSpeed。

已有 PyTorch 基础保留，当前从 Multi-Head Attention、RoPE 和 tensor layout 继续。阶段规划见 [MiniLM 学习规划](Mini-LM/Mini-LM.md)。

## 2. cpp-llm-engine：C++ 推理引擎

先按推理引擎需求补充现代 C++ 基础与 C++17 特性，包括引用与指针、对象生命周期、RAII、智能指针、移动语义、容器和必要的构建调试知识；此时不展开完整 CS106L 课程。

按照 Datawhale 教程推进模型加载、Tensor 与算子、Decoder、生成与 KV Cache、调度和服务。复用此前 llama.cpp 学习成果作为基线与对照，按教程的 CPU 六任务要求完成学习和验收。

## 3. zero-to-sglang：推理框架实践

按照 Datawhale 教程完成推理基础、Mini SGL 实践和真实 SGLang 源码学习，理解请求生命周期、缓存、调度、并行与性能分析。

保留教程要求的动手环节，并与 cpp-llm-engine、此前的 llama.cpp 学习进行对照。具体章节任务与验收在本阶段目录维护，开源贡献作为选做内容。

## 4. CS106L：系统掌握现代 C++

在前两个推理项目的实践基础上，系统学习 CS106L，整理类型、泛型、资源管理、值语义及标准库知识，提高现代 C++ 的理解与熟练度。

完成所选学期的课程练习与作业，并回顾推理项目中的设计。LeetCode 与实习面试准备放到这一阶段再启动，近期不并行安排。

## 学习方式与范围

- 采用“中文直觉与英文术语 → 数学与数据流 → API 小例子 → 独立实现 → 测试与 review”的方式学习。
- Agent 负责指导、练习设计、错误分析和代码审阅；学习代码由学习者编写，Agent 不代写实现或直接修复代码。
- 每阶段保留代码、测试或基线对照、中文记录；详细验收由对应目录维护。
- 本地以 CPU、小规模正确性实验为主，GPU 训练和框架性能实验按需使用服务器。
- CS336 正式作业独立完成，遵守各作业目录的学术诚信规约。
- Kernel 优化按实际问题选学；Agentic RL 保留为远期扩展，暂不安排。

## 主要资料参考

- [Diy-LLM](https://datawhalechina.github.io/diy-llm/)
- [CS336](https://cs336.stanford.edu/)
- [MiniMind](https://github.com/jingyaogong/minimind)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [cpp-llm-engine CPU 六任务计划](https://github.com/datawhalechina/cpp-llm-engine/blob/main/source/guides/cpu_learning_plan.md)
- [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang/blob/main/README_zh.md)
- [CS106L](https://web.stanford.edu/class/cs106l/)
