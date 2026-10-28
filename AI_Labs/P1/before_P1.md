# Transformer

Attention is all you need !

## embedding

语言模型主要的数据流 : 文本 → Tokenizer → input_ids [B, T] → Embedding → hidden states [B, T, C] → Transformer Blocks → logits [B, T, V]\
这里B(batch size) ; T(序列长度) ; V(词表大小) ; C(hidding size/embedding dimension)

什么是"词表"?

- 模型认识多少种不同的token,在LLM中，一般可以将词表看作一个包含了"所有可用token"以及"每种token对应的token ID"组成的双向映射集合

词表是哪里来的?

- 收集大量语料(如来自网页、代码、论文等)，但是这样会出现语料库中没有的词模型就不能识别的情况，所以现在LLM会从基础单位开始(字符、Unicode字符等)，之后通过算法进行合并(如BPE,假如一个词出现的频率高,就会将其合并),这样即使模型遇到之前没见过的词也可以用现有组合拼出新词
- 一个LLM的词表一般包含:普通文本(如中英文)、字符字节、标点(空格+A和A是不一样的),特殊token,如:

```text
<bos>           # 序列开始
<eos>           # 序列结束
<pad>           # 补齐序列
<unk>           # 未知内容
<|user|>        # 用户消息开始
<|assistant|>   # AI回复开始
```

embedding的本质是用一个token ID，从可训练矩阵中取出对应的一行，这个可训练矩阵的`shape=[V,d]`,每一行表示一个token,列表示embedding dimension\
embedding查表时通过检索对应token ID的`embedding_matrix[token_id]`\
在真实处理时，一段话先被转换成其token ID(在词库中的位置,shape是一个一维tensor $token\_ids = [x_1,x_2,…,x_V]$ , $x_i$ 表示这个token在词库中的id),embedding查表的过程就是找到行数为i的token

常用api

```python
nn.Embedding(
    num_embeddings=V,
    embedding_dim=C,
    padding_idx=None,
    sparse=False,
)

token_ids = torch.tensor([x_1, x_2, x_V])
output = embedding(token_ids)
```

输出:行表示第token_id个token,列表示embedding dimension\
上面说的其实是当batch size = 1时的情况，一般token_ids不会是一个一维张量，而是shape=[B,T]的张量，但是原理相同

如何训练embedding矩阵?

- 初始时embedding矩阵的d(embedding dimension)是人为选择的，这将直接影响模型的参数量
- 一个简单的完整的embedding更新示例:先查表得到 Token ID 对应的 \(h\)，根据 \(z=hW\) 计算 logits，经过 softmax 得到概率，和真实 one-hot 对比，然后反向传播。(代码示例如下)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

V = 3  # 词表大小
d = 2  # Embedding dimension

# E: [V, d]，随机初始化
embedding = nn.Embedding(V, d)

# PyTorch 保存为 [V, d]，数学计算相当于 hW
lm_head = nn.Linear(d, V, bias=False)

# Token ID 0 作为输入
input_id = torch.tensor([0])

# 正确的下一个 Token 是 ID 1
target = torch.tensor([1])

# 观察训练前参数
print("训练前的 E:")
print(embedding.weight)
print("训练前的输出层 W:")
print(lm_head.weight)

# 1. 查表
h = embedding(input_id)
print("h shape:", h.shape)  # [1, 2]

# 2. 计算 logits
logits = lm_head(h)
print("logits:", logits)
print("logits shape:", logits.shape)  # [1, 3]

# 3. 查看 softmax 概率
probabilities = F.softmax(logits, dim=-1)
print("probabilities:", probabilities)

# 4. 交叉熵损失
loss = F.cross_entropy(logits, target)
print("loss:", loss.item())

# 5. 反向传播
loss.backward()

print("Embedding 的梯度:")
print(embedding.weight.grad)
print("输出层的梯度:")
print(lm_head.weight.grad)
```

在真实LLM中,不使用上面的$h=E[x], z=hW$,而是得到embedding后经过多层transformer\
实现了一个简单的embedding,在`AI_Labs\P1\embedding_Labs.py`

## position

embedding只解决了让文字能被模型认识，对应到token，但是没有得到"这个token在什么位置"\
虽然训练时Decoder的causal mask(单向注意力掩码:在训练阶段，本身需要模型预测下一个token，如果不进行掩码，模型将能看到完整的token序列，训练就不起作用了)对位置本身有一定顺序，但是还需要专门的位置机制\
常用方法：可学习绝对位置嵌入；固定正余弦位置编码；相对位置编码；RoPE(现代 LLM 最常见的方案之一)

### 可学习位置嵌入

专门为position创建一个embedding :\
token embedding的shape=[V,C] ; position embedding的shape=[max_seq_len,C]\
对位置t : $x[b,t] = token\_embedding[token_id] + position\_embedding[t]$
