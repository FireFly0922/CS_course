import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ManualMultiHeadCausalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )

        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )

    def split_heads(self, x):
        # x: [B, T, C]
        B, T, C = x.shape

        # [B, T, C] → [B, T, H, D]
        x = x.reshape(
            B,
            T,
            self.num_heads,
            self.head_dim,
        )

        # [B, T, H, D] → [B, H, T, D]
        x = x.transpose(1, 2)

        return x

    def merge_heads(self, x):
        # x: [B, H, T, D]
        B, H, T, D = x.shape

        # [B, H, T, D] → [B, T, H, D]
        x = x.transpose(1, 2)

        # [B, T, H, D] → [B, T, C]
        x = x.contiguous().view(B, T, H * D)

        return x

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape

        # 1. 线性投影
        q = self.q_proj(x)  # [B, T, C]
        k = self.k_proj(x)  # [B, T, C]
        v = self.v_proj(x)  # [B, T, C]

        # 2. 拆分 head
        q = self.split_heads(q)  # [B, H, T, D]
        k = self.split_heads(k)  # [B, H, T, D]
        v = self.split_heads(v)  # [B, H, T, D]
# 这里是第一种写法便于理解，下面还有更工程的封装写法
        # 3. 每个 head 独立计算 Attention score
        scores = q @ k.transpose(-2, -1)

        # scores: [B, H, T, T]
        scores = scores / math.sqrt(self.head_dim)

        # 4. Causal mask
        causal_mask = torch.tril(
            torch.ones(
                T,
                T,
                dtype=torch.bool,
                device=x.device,
            )
        )

        scores = scores.masked_fill(
            ~causal_mask,
            float("-inf"),
        )

        # 5. Attention probability
        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        # 6. 每个 head 分别聚合 V
        head_output = attention_weights @ v

        # head_output: [B, H, T, D]

        # 7. 合并 heads
        output = self.merge_heads(head_output)

        # output: [B, T, C]

        # 8. 输出投影，让不同 head 的信息重新混合
        output = self.out_proj(output)

        return output, attention_weights
"""
        # 这里展示工程化写法，可以替换上面的部分
        dropout_p = {
            self.dropout_p
            if self.training
            else 0.0
        }
        head_output = F.scaled_dot_product_attention(q,k,v,dropout_p = dropout_p, is_causal=True)
        output = self.merge_heads(self.head_output)
        output = self.out_proj(output)
        return output
"""


torch.manual_seed(42)

x = torch.randn(2, 4, 8)

attention = ManualMultiHeadCausalAttention(
    embed_dim=8,
    num_heads=2,
)

output, weights = attention(x)

print("input:", x.shape)
print("output:", output.shape)
print("weights:", weights.shape)