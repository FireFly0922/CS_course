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
在真实处理时，一段长度为 $T$ 的文本先被转换成一维tensor $token\_ids = [x_1,x_2,…,x_T]$，其中 $x_t$ 表示第 $t$ 个位置的token ID；embedding查表就是取出`embedding_matrix[x_t]`

常用api

```python
nn.Embedding(
    num_embeddings=V,
    embedding_dim=C,
    padding_idx=None,
    sparse=False,
)

token_ids = torch.tensor([x_1, x_2, x_T])
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
虽然训练时Decoder的causal mask规定了“只能看当前位置及之前的位置”，引入了方向约束，但它不能为每个token提供明确的绝对或相对位置信息，所以仍需要专门的位置机制\
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

- $q_i^{\top}k_j$

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
        scores = q @ k.transpose(-2,-1)    # [B,T,C] @ [B,C,T] → [B,T,T]
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
将[B, T, C]reshape成[B, T, H, D]，SDPA(缩放点积注意力)希望接收[B, H, T, D], 需要transpose\
便于理解的完整MHA(Multi-Head Attention)实现在`AI_Labs\P1\MHA.py`\
另外，一般定义split_heads和merge_heads, 定义merge_heads时需要contiguous将内存连续, 输出时需要merge_heads

这里只展示封装最完善的`nn.MultiheadAttention`版本实现，另外两种自定义性较强的版本在`AI_Labs\P1\MHA.py`\
这种写法是最方便的，但是可能不太方便自定义 RoPE、KV Cache、GQA 等

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

B, T, C = 2, 4, 8
H = 2

x = torch.randn(B, T, C)

mha = nn.MultiheadAttention(
    embed_dim=C,
    num_heads=H,
    batch_first=True,
)

# nn.MultiheadAttention 中：
# True 表示该位置要被屏蔽
causal_mask = torch.triu(
    torch.ones(T, T, dtype=torch.bool),
    diagonal=1,
)

output, weights = mha(
    query=x,
    key=x,
    value=x,
    attn_mask=causal_mask,
    need_weights=True,
    average_attn_weights=False,
)

print("output:", output.shape)
print("weights:", weights.shape)
```

## 旋转位置编码(RoPE)

之前学过绝对位置编码，RoPE也是一种让Attention理解词语"位置"的方法\
先进行数学推导：\
$R$指对某位置token的旋转矩阵
$$q_m' = R_m q_m ; k_n' = R_n k_n$$
$$(q_m')^{\top} k_n' = q_m^{\top} R_m^{\top} R_n k_n = q_m^{\top} R_{n-m} k_n$$
旋转并不会改变向量的模长。上式说明Attention点积中的位置旋转项只与$n-m$这一相对距离有关；完整点积仍然同时依赖$q_m$和$k_n$的内容\
实现的RoPE在`AI_Labs\P1\RotaryEmbedding.py`，后面学习完这些概念后会实现一个完整的Transformer

## Transformer的其它组件与连接方式

### LayerNorm

对Transformer输入`[B,T,C]`中的每个token，沿最后一个隐藏维度`C`独立归一化，用于稳定训练过程
$$\text{LayerNorm}(v) = \gamma \frac{v-\mu}{\sigma} + \beta$$
这里计算均值和标准差是对同一层的所有神经元进行计算，另外计算标准差时$\sigma = \sqrt{\frac{1}{d} \sum_{i=1}^d(v_i - \mu)^2 + \epsilon}$中的$\epsilon$是一个极小量防止计算归一化公式时出现0

这里$\gamma$和$\beta$是可学习参数

### 残差连接

这种方法可以让信息跳过某些层传递。残差分支学习$y = F(x) + x$；即使$F$暂时学不到有效变换，模型仍然可以通过$x$保留恒等映射。

现代LLM常用$x_{new} = x + F(Norm(x))$的`Pre-Norm`形式残差路径。若$y=x+F(x)$，则$\frac{\partial\mathcal L}{\partial x}=\frac{\partial\mathcal L}{\partial y}\left(I+\frac{\partial F(x)}{\partial x}\right)$，其中$I$提供直接的梯度通路

```python
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        attention,
        ffn,
    ):
        super().__init__()
        self.attention_norm = nn.RMSNorm(
            embed_dim,
            eps=1e-6,
        )
        self.attention = attention
        self.ffn_norm = nn.RMSNorm(
            embed_dim,
            eps=1e-6,
        )
        self.ffn = ffn
    def forward(
        self,
        x,
        start_pos=0,
    ):
        # 第一个残差分支
        residual = x
        normalized = self.attention_norm(x)
        attention_output = self.attention(
            normalized,
            start_pos=start_pos,
        )
        x = residual + attention_output
        # 第二个残差分支
        residual = x
        normalized = self.ffn_norm(x)
        ffn_output = self.ffn(normalized)
        x = residual + ffn_output
        return x
```

### RMSNorm

对输入的[B, T, C]的最后一维C进行独立的归一化
$$\operatorname{RMS}(x) = \sqrt{\frac{1}{C}\sum_{i=1}^{C}x_i^2+\epsilon}$$
$$\operatorname{RMSNorm}(x) = \gamma\odot\frac{x}{\operatorname{RMS}(x)}$$

在写代码时，一般使用的$$\operatorname{normalized}(x_i) = x_i\cdot\frac{1}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\epsilon}}$$
完整RMSNorm$$\operatorname{RMSNorm}(x) = \gamma\odot\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}$$

手写版本的RMSNorm

```python
class ManualRMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        # 每个隐藏维度都有一个可训练缩放值
        self.weight = nn.Parameter(torch.ones(dim))
        # 这里是创建一个会被模型训练更新的参数向量

    def forward(self, x):
        # x: [B, T, C]
        input_dtype = x.dtype
        # 归一化统计通常在 float32 中计算
        x_float = x.float()
        mean_square = (x_float.pow(2).mean(dim=-1, keepdim=True)) # pow(2)是将其平方
        inverse_rms = torch.rsqrt(mean_square + self.eps) # rsqrt表示$\frac{1}{\sqrt{x}}$
        normalized = x_float * inverse_rms
        normalized = normalized.to(dtype=input_dtype)
        # weight: [C]
        # 自动广播到 [B,T,C]
        return normalized * self.weight
```

pytorch提供简单的api封装版本`nn.RMSNorm(normalized_shape, eps=None, elementwise_affine=True)`

```python
norm = nn.RMSNorm(normalized_shape=8, eps=1e-6)
x = torch.randn(2,3,8)
y = norm(x)
```

上面手写版的RMSNorm中，为什么要转换成float32 ?\
用低精度保存和计算，但是敏感的操作还是需要使用高精度，如“平方和、均值、倒数平方根等统计操作”，这样的操作比如还有：RMSNorm/LayerNorm中的统计量；softmax中的内部运算等

### FFN层

在transformer中，FFN(前馈网络)负责对每个token内部特征进行非线性加工, 其在transformer block中一般先让Attention混合不同token, 之后将混合了不同token的结果交给FFN层\
经典的Transformer FFN公式$$\text{FFN}(x) = \text{ReLU}(x W_1^{\top} + b_1) W_2^{\top} + b_2$$
这里还需要了解上投影(通过线性层让维数从`C`变为更大的`F`, eg:[B,T,C]→[B,T,F])和下投影(从`F`恢复为`C`, [B,T,F]→[B,T,C])，下面展示一个典型的FFN

```python
class FeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.up_proj = nn.Linear(embed_dim,hidden_dim)
        self.down_proj = nn.Linear(hidden_dim,embed_dim)

    def forward(self, x):
        # x: [B,T,C]
        hidden = self.up_proj(x)
        # [B,T,C] → [B,T,F]
        hidden = F.relu(hidden)
        # [B,T,F]，shape 不变
        output = self.down_proj(hidden)
        # [B,T,F] → [B,T,C]
        return output
```

### SiLU/Swish

定义$\text{SiLU}(x) = x \sigma(x)$，这里的$\sigma$是sigmoid而不是softmax，即$\sigma(x) = \frac{1}{1+e^{-x}}$；在`torch.nn.functional`中调用`F.silu(x)`等同于`x * torch.sigmoid(x)`

### SwiGLU

对比普通FFN, 现代Decoder only的LLM大多使用SwiGLU作为FFN/MLP, 其在两条投影上有分支

```text
                  ┌→ gate_proj → SiLU ─┐
x ────────────────┤                    ├→ 逐元素乘法 → down_proj
                  └→ up_proj ──────────┘
```

上面的gate_proj是门控，用于控制信息通过多少

```python
class SwiGLU(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim,hidden_dim,bias=False)
        self.up_proj = nn.Linear(embed_dim,hidden_dim,bias=False)
        self.down_proj = nn.Linear(hidden_dim,embed_dim,bias=False)

    def forward(self, x):
        gate_logits = self.gate_proj(x)
        gate = F.silu(gate_logits)
        value = self.up_proj(x)
        hidden = gate * value
        output = self.down_proj(hidden)
        return output
```

在数学上$\left[\operatorname{SiLU}(xW_g^\mathsf T) \odot (xW_u^\mathsf T)\right]W_d^\mathsf T$, 可以拆开来看 : 门控信号$x W_g^{\top}$对当前token生成一组调制值 ; 另外的Value分支$x W_u^{\top}$负责产生候选内容，可以理解为对某个中间维度j, Value分支决定提供特征内容, gate分支决定如何修改这个feature

## 完整Decoder-only Transformer前向

前面学习了很多transformer的组件，下面先梳理一下这些组件是如何连接的，数据流如何流动: \
token_ids [B,T] → token embedding → [B, T, C] → n个transformer block → final RMSNorm [B, T, C] → LM head → logits [B, T, V] → Cross-Entropy Loss

### token ID 到 [B,T,C]

通过Embedding查表将原本[B, T]的token_id映射为[B, T, C]，后面的Transformer都在处理这个隐藏维度C\
值得注意的是一般不在这里进行RoPE

### Transformer Block

分别展示两条数据流，但是这里说的可能不太清楚，还是看代码直观：

- RMSNorm → Attention :\
先对每个token归一化，这一步中shape不变；Attention经过三个linear生成shape为[B, T, C]的QKV，再reshape为[B, T, H, D]并transpose为[B, H, T, D]。RoPE作用于这个layout下的Q和K，之后计算Attention、合并heads、输出投影，并与进入该分支前的residual相加。

- RMSNorm → SwiGLU :\
先对Attention残差相加后的[B, T, C]再次进行RMSNorm，再经过SwiGLU的`C → F → C`变换，最后与进入FFN分支前的residual相加。

### LM Head(Language Modeling Head语言输出头)

经过了Transformer block后,由于LLM最终的问题是“预测下一个token”, 需要对当前的[B, T, C]进行一些处理\
LM Head其实只是一个线性层$\text{logits}_{b,t,:} = x_{b,t} W^{\top}$，其中$W$的shape为[V,C]，输出logits的shape为[B,T,V]。$\text{logits}_{b,t,v}$可以理解为当前隐藏状态和词表中第$v$个输出向量的点积，点积越大，模型越倾向于预测该token。logits是未归一化分数，训练时直接传给Cross-Entropy，不要提前softmax。

Embedding权重和LM Head权重的shape都是[V,C]，可以通过`lm_head.weight = token_embedding.weight`进行weight tying，从而减少参数量。

### Next-token Prediction与Cross-Entropy

语言模型学习的是预测下一个token。例如原始序列为`[BOS, 我, 喜欢, 猫, EOS]`，则输入为`[BOS, 我, 喜欢, 猫]`，targets为`[我, 喜欢, 猫, EOS]`。causal mask负责阻止模型看到未来输入，shifted targets负责规定每个位置应该预测的下一个token，两者职责不同。

```python
input_ids = tokens[:, :-1]   # [B,T]
targets = tokens[:, 1:]      # [B,T]，dtype必须是torch.long

logits = model(input_ids)    # [B,T,V]
loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1),
    ignore_index=-100,
)
```

如果数据集已经构造了错开一位的`input_ids`和`targets`，模型内部就不能再次shift。`ignore_index=-100`可用于忽略padding或SFT中不参与训练的位置。
