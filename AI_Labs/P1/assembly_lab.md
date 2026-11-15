# Tiny Transformer 拼装练习

本练习位于 Transformer Foundations 到最小训练闭环的衔接处。重点是模块组合、参数注册、shape 和数据/标签流；已有旋转算法与残差实现可以复用，不涉及正式 CS336 作业。

状态：已提供学习用 TODO 和接口约定。先独立完成 assembly_lab.py，再提交代码 review；当前不安排独立测试脚本，尚未完成验收。

## 从哪个文件开始

- 在 `assembly_lab.py` 填写 TODO A—F；每个 TODO 完成后删除对应的占位异常。
- `Tiny_transformer.py` 中你已完成的 `TransformerBlock` 被直接导入，不必再抄写一遍。
- `Tiny_transformer_start.py` 是旧的 Block 模板，本练习不使用它。

TODO 0：将 `RotaryEmbedding.py` 底部从 `torch.manual_seed(42)` 开始的演示移到 `if __name__ == "__main__":` 内并缩进。否则导入这个类会执行演示、打印输出并重置全局随机种子。类定义保持在外面。这是你需要完成的一项工程练习。

TODO 0 只处理模块导入时的副作用，不要求编写或运行冒烟测试脚本。

## 谁创建谁

```text
调用者持有 ModelConfig
└─ TinyDecoderLM(config)
   ├─ token_embedding
   ├─ blocks: ModuleList
   │  └─ build_block(config)，每层调用一次
   │     └─ 已完成的 TransformerBlock
   │        ├─ attention_norm / ffn_norm（原 Block 已创建）
   │        ├─ CausalSelfAttention(config)
   │        │  ├─ q_proj / k_proj / v_proj / out_proj
   │        │  └─ RotaryEmbedding
   │        └─ SwiGLU(config)
   ├─ final_norm
   └─ lm_head（与 token_embedding 共享 weight）

训练调用者还持有 optimizer；它在循环外创建一次。
```

`__init__` 创建并注册需要长期保存的模块/参数；`forward` 使用这些模块处理当前输入。不要在 forward 内重新创建 Linear。将模块实例赋给 self 的属性后，PyTorch 才能通过父模块管理它的参数。

## 接口约定

| 名称 | 输入 | 返回 | 所属 TODO |
| --- | --- | --- | --- |
| SwiGLU | float [B,T,C] | float [B,T,C] | A |
| CausalSelfAttention | float [B,T,C], start_pos | 单个 float [B,T,C] | B |
| build_block | config | 新的 TransformerBlock 实例 | C |
| TinyDecoderLM | long [B,T], 可选 targets | (logits [B,T,V], loss 或 None) | D |
| make_batch | long [B,T+1] | (input_ids [B,T], targets [B,T]) | E |
| train_step | model, optimizer, inputs, targets | loss/grad_norm 的 Python float 字典 | F |

第一版固定 CPU float32、dropout=0，无 padding 输入、无 KV Cache。targets 允许 -100，但至少有一个有效目标。正常 ID 必须位于 [0,V)，模型和数据必须在同一 device。默认 C=32、H=4、D=8、F=64。

## 按顺序完成

1. A：连接三条 FFN 投影。检查逐 token 独立性。
2. B：借用 MHA.py 的投影/拆头/合头知识，在计算注意力之前连接已有 RoPE，并统一为 Tensor 返回值。旧 MHA 的二元组不能直接加到 residual 上。
3. C：把 A/B 的实例传给已完成的 TransformerBlock。这里练习的是对象组合，不重写残差算法。
4. D：通过 ModuleList 堆叠独立 Block，连接 Embedding、Final RMSNorm、LM Head；实现有无 targets 两种返回。
5. E：从一条序列构造输入与下一 token 标签；只在这一处 shift。
6. F：连接训练的一次更新；optimizer 必须在循环外实例化。

C 依赖 A+B，D 依赖 A+B+C，E 独立，F 依赖 D+E。每项实现完成后删除对应的 NotImplementedError 占位语句；它不是输入校验。完成整个文件后，再提交代码 review。

## 需要自己解释的拼装问题

- Block 为什么接收模块实例，而不是模块的输出 Tensor？
- 每层为什么需要新建 Attention/FFN，不能把同一个 Block 重复放 N 次？
- Attention 返回 tuple 时为什么不能直接做 residual add？
- start_pos 传到哪里？为什么它不能替代历史 KV Cache？
- loss 为什么必须保留 Tensor 直到 backward，日志却应该转成 float？
- state_dict 为什么能从顶层找到内层 Linear 参数和 RoPE buffers？

## 验收边界

完成后先做静态代码 review，检查接口、shape、独立参数、RoPE 调用、因果性、残差和训练步骤顺序。静态 review 不等于运行验收；后续再安排必要的数值对照、单 batch 过拟合、保存加载一致性和中文实验记录，不在当前学习目录提前铺设测试脚本。只有阶段 F 有一次 optimizer 更新，不含完整训练循环。

后续由你独立完成一个固定 batch 的训练入口和 loss 记录。这个完整模型会作为将来 KV Cache 与 C++ forward 的数值基线。
