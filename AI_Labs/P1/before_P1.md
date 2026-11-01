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
对位置t : $x[b,t] = token\_embedding[token\_id] + position\_embedding[t]$

```python
class TokenPositionEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=d_model,
        )

    def forward(self, input_ids):
        # input_ids: [B, T]

        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )

        position_ids = torch.arange(
            seq_len,
            device=input_ids.device,
        )
        # [T]

        token_x = self.token_embedding(input_ids)
        # [B, T, C]

        position_x = self.position_embedding(position_ids)
        # [T, C]

        x = token_x + position_x
        # [B, T, C] + [T, C]
        # position_x 会在 batch 维广播

        return x
```

### 正余弦位置编码

$$PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/C}}\right)$$
$$PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/C}}\right)$$
也就是不学习位置矩阵 : pos表示token位置 ; i表示维度对的编号

```python
class SinusoidalPositionEncoding(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        d_model: int,
    ):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError("d_model must be even")

        # [T, 1]
        position = torch.arange(max_seq_len,dtype=torch.float32).unsqueeze(1)

        # [C / 2]
        div_term = torch.exp(
            torch.arange(0,d_model,2,dtype=torch.float32) * (-math.log(10000.0) / d_model)
            )

        pe = torch.zeros(max_seq_len,d_model)

        # 偶数维
        pe[:, 0::2] = torch.sin(position * div_term)   # 切片操作，表示：所有行维度，列维度从0开始每次+2

        # 奇数维
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe",pe)   # 下面会讲解其和parameter的区别

    def forward(
        self,
        x,
        start_pos: int = 0,
    ):
        # x: [B, T, C]

        seq_len = x.shape[1]
        end_pos = start_pos + seq_len

        position_x = self.pe[start_pos:end_pos]
        # [T, C]

        position_x = position_x.to(dtype=x.dtype)

        return x + position_x.unsqueeze(0)
```

`nn.Parameter`通常`requires_grad = True`，会交给优化器更新参数\
`register_buffer`不是可训练参数，会随着model.to(device)搬运

## 单头 Scaled Dot-Product Attention

### Attention 公式的理解

对一个token,在key、Value、Query三个维度有不同的映射，"映射"这一过程，是通过训练三个矩阵得到的
$$\text{Attention}(Q,K,V) = \text{softmax} ( \frac{QK^{\top}}{\sqrt{d_k}} + M )V$$

- $q_i^{\top}j_j$

通过点积计算两向量的匹配度$$q_i^{\top}k_j = \sum_{r=1}^{d_k} q_{ir} k_{jr}$$
也就是说每个q都要和所有k比较，从而计算出第i个token认为第j个token有多相关

- 为什么缩放系数和$d_k$有关

在计算$QK^{\top}$时，$
Q=\begin{bmatrix}
q_{11} & q_{12} & q_{13}\\
q_{21} & q_{22} & q_{23} 
\end{bmatrix}$, $
K^\mathsf T=
\begin{bmatrix}
k_{11} & k_{21}\\
k_{12} & k_{22}\\
k_{13} & k_{23}
\end{bmatrix}$

$QK^\mathsf T=
\begin{bmatrix}
q_{11}k_{11}+q_{12}k_{12}+q_{13}k_{13} & q_{11}k_{21}+q_{12}k_{22}+q_{13}k_{23}
\\
q_{21}k_{11}+q_{22}k_{12}+q_{23}k_{13} & q_{21}k_{21}+q_{22}k_{22}+q_{23}k_{23}
\end{bmatrix}$

随着d增大, $QK^{\top}$矩阵中每一元素也会增大，也就是说缩放系数和d有关(这里的d指KQV的列数)

- 为什么除以的是$\sqrt{d_k}$, 而不是$d_k$

我们的目标是让Attention logits中的典型数值范围保持基本不变\
$QK^{\top}$矩阵的方差随$d_k$线性增长，所以实际数值尺度(标准差)与$\sqrt{d_k}$成正比

### 完整单头缩放点积注意力的组成

简单来说就是 : 生成QKV → 计算QK相关性 → 缩放 → Causal mask → 对K维做softmax → 聚合V

```python
class SingleHeadCausalAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = q @ k.transpose(0,1)      # QK相关性
        scores = scores / math.sqrt(C)     # 缩放
        # 下面会讲解这2句代码是怎么实现因果掩码的
        mask = torch.tril(torch.ones(T,T,device=x.device,dtype=torch.bool)) 
        scores = scores.masked_fill(~mask, float("-inf"))
        atten_weights = torch.softmax(scores, dim=-1) # 对k维度做softmax求概率
        output = atten_weights @ v # 聚合V(套公式,此时shape为[B,T,C])
        return output, atten_weights
```

- attention的核心"因果掩码"

`torch.ones(T,T,device=x.device,dtype=torch.bool)`先创建一个全为1,shape=[T,T]的，并将其中的1转换成bool格式\
`mask = torch.tril(…)`取下三角矩阵置1，其他部分置0，也就是False\
`scores = scores.masked_fill(~mask, float("-inf"))`先对mask取反，因为masked_fill填充的是True的部分，将上三角区域填充为`-inf`，这是因为softmax计算时会对元素取e的指数，$e^{-inf}$可以得到0\
这样实现了求第a个token的Attention时只能看到前a个token，防止提前看到正确答案

项目中一般使用pytorch中已有的Attention的API`torch.nn.functional.scaled_dot_product_attention`

```python
torch.nn.functional.scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
)
```

由于涉及了一些新概念，在下一节中会详细讲解

## Multi-Head Attention

多头注意力(简称MHA)将隐藏维度切成多个子空间，可以理解为从多个角度学习问题\
将[B, T, C]reshape成[B, T, H, D]，SDPA(缩放点积注意力)希望接收[B, H, T, D]
