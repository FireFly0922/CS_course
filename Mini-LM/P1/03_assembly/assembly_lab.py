"""TinyLM 拼装练习。阅读 assembly_lab.md，依次完成 TODO A—F。

仅针对自建学习项目。CPU / float32 / dropout=0 / 无 KV Cache。
原有 TransformerBlock 已实现，在这里直接复用。
本文件不在 import 时训练、创建模型或修改随机种子。
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from Tiny_transformer import TransformerBlock
from RotaryEmbedding import RotaryEmbedding as RoPE

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16
    max_seq_len: int = 16
    embed_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    hidden_dim: int = 64
    norm_eps: float = 1e-6
    rope_base: float = 10000.0

    def __post_init__(self):
        for name in (
            "vocab_size", "max_seq_len", "embed_dim", "num_layers",
            "num_heads", "hidden_dim",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head_dim")
        if self.norm_eps <= 0 or self.rope_base <= 0:
            raise ValueError("norm_eps and rope_base must be positive")

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # TODO A1：创建 gate_proj、up_proj、down_proj，全部 bias=False。
        # 前两者 C→F，最后一个 F→C。复用笔记中的知识即可。
        self.gate_proj = nn.Linear(config.embed_dim,config.hidden_dim,bias=False)
        self.up_proj = nn.Linear(config.embed_dim,config.hidden_dim,bias=False)
        self.down_proj = nn.Linear(config.hidden_dim,config.embed_dim,bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO A2：输入 [B,T,C]，两分支中间结果 [B,T,F]，输出 [B,T,C]。
        # SiLU 作用于 gate 分支；两分支按元素相乘，再下投影。
        gate_logits = self.gate_proj(x)
        gate = F.silu(gate_logits)
        value = self.up_proj(x)
        hidden = gate * value
        output = self.down_proj(hidden)
        return output

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, dropout_p: float = 0.0):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len
        self.rope_base = config.rope_base
        self.dropout_p = float(dropout_p)
        self.config = config
        # TODO B1：创建 q_proj、k_proj、v_proj、out_proj（C→C，无 bias）。
        # 创建 self.rope，使用已有 RotaryEmbedding 类，不重写旋转算法。
        # 导入前先完成指南中的 TODO 0，避免其演示代码在 import 时执行。
        # 可在此处导入：from RotaryEmbedding import RotaryEmbedding
        # RoPE 构造参数取 config.head_dim / max_seq_len / rope_base。
        C = config.embed_dim
        self.q_proj = nn.Linear(C, C, bias = False)
        self.k_proj = nn.Linear(C, C, bias = False)
        self.v_proj = nn.Linear(C, C, bias = False)
        self.out_proj = nn.Linear(C, C, bias = False)
        self.rope = RoPE(head_dim=self.head_dim, max_seq_len=self.max_seq_len, base=self.rope_base)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # TODO B2：复用你已掌握的拆头过程。
        # [B,T,C] → [B,T,H,D] → [B,H,T,D]。
        # reshape 拆分 C；transpose 交换维度，通常只改变 stride。
        B, T, C = x.shape
        x = x.reshape(B, T, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)
        return x

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # TODO B3：[B,H,T,D] → [B,T,H,D] → [B,T,C]。
        # contiguous 可能复制数据，或用允许必要复制的 reshape。
        B, H, T, D = x.shape
        x = x.transpose(1, 2)
        x = x.reshape(B, T, self.config.embed_dim)
        return x

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.config.embed_dim:
            raise ValueError("expected x with shape [B,T,embed_dim]")
        if start_pos < 0 or start_pos + x.size(1) > self.config.max_seq_len:
            raise ValueError("position outside RoPE cache")
        # TODO B4：连接已有零件，不返回 attention_weights。
        # x → 三个投影 → split_heads → RoPE(Q,K) → SDPA
        #   → merge_heads → out_proj → 返回单个 [B,T,C] Tensor。
        # V 不旋转；把 start_pos 传给 RoPE。
        # 本练习 Q/K 序列等长、没有历史缓存，SDPA 使用
        # dropout_p=0.0、is_causal=True，不要重复手动缩放 Q/K。
        # 注意：有 start_pos 并不意味着已实现 KV Cache。
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))
        q,k = self.rope(q, k, start_pos=start_pos)
        atten_dropout = self.dropout_p if self.training else 0.0
        head_output = F.scaled_dot_product_attention(query=q,key=k,value=v,dropout_p=atten_dropout,is_causal=True)
        output = self.merge_heads(head_output)
        output = self.out_proj(output)
        return output

def build_block(config: ModelConfig) -> TransformerBlock:
    # TODO C：创建一个 CausalSelfAttention 实例和一个 SwiGLU 实例，
    # 注入你已经完成的 TransformerBlock，传入 embed_dim/norm_eps。
    # 返回实例。每次调用必须创建独立的子模块，不能共享上一层参数。
    atten = CausalSelfAttention(config=config,dropout_p=0.0)
    ffn = SwiGLU(config=config)
    block = TransformerBlock(embed_dim=config.embed_dim, attention=atten, ffn=ffn, norm_eps=config.norm_eps)
    return block

class TinyDecoderLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # TODO D1：注册四部分：
        # token_embedding：V→C 的查表；blocks：nn.ModuleList，含 N 个独立 block；
        # final_norm：nn.RMSNorm(C, eps=...)；lm_head：Linear(C,V,bias=False)。
        # 通过 build_block(config) 创建每层，不使用 [同一个block] * N。
        # 最后让 lm_head.weight 与 token_embedding.weight 共享同一 Parameter。
        self.token_embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embed_dim)
        self.blocks = nn.ModuleList([
            build_block(config=config)
            for _ in range(config.num_layers) 
            ])
        self.final_norm = nn.RMSNorm(normalized_shape=config.embed_dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(in_features=config.embed_dim, out_features=config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be long [B,T]")
        if not 1 <= input_ids.size(1) <= self.config.max_seq_len:
            raise ValueError("sequence length outside configured range")
        if targets is not None:
            if targets.shape != input_ids.shape or targets.dtype != torch.long:
                raise ValueError("targets must be long with the same [B,T] shape")
            if not (targets != -100).any():
                raise ValueError("at least one supervised target is required")
        # TODO D2：input_ids → embedding → blocks（逐层，start_pos=0）
        # → final_norm → lm_head → logits [B,T,V]。返回 raw logits。
        # TODO D3：targets=None 时返回 (logits, None)；否则展平后计算 CE。
        # targets 已由数据端错开一位，这里不能再次 shift。
        # 使用 ignore_index=-100。loss 为 scalar Tensor，不能提前 .item()。
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x, start_pos = 0)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        flat_logits = logits.reshape(-1, self.config.vocab_size)
        flat_targets = targets.reshape(-1)
        loss = F.cross_entropy(input=flat_logits, target=flat_targets, ignore_index=-100)
        return logits, loss

def make_batch(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """输入 long [B,T+1]，返回同 device 的 input_ids/targets，各为 [B,T]。"""
    if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.size(1) < 2:
        raise ValueError("tokens must be long [B,T+1] with at least two positions")
    # TODO E：前后错开一位。切片可共享存储且不一定连续，后续用 reshape。
    without_last = tokens[:, :-1]
    without_first = tokens[:, 1:]
    return without_last, without_first

def train_step(
    model: TinyDecoderLM,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    # TODO F：train 模式 → 清梯度 → forward → backward → 裁剪 → step。
    # optimizer 从外部传入；不能在这里重新创建。
    # clip_grad_norm_ 使用 max_norm=1.0 / error_if_nonfinite=True。
    # 返回 {"loss": Python float, "grad_norm": Python float}。
    # loss 是本次更新前的 loss，grad_norm 是裁剪前的总范数。
    
    # 训练
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(input_ids, targets)
    if loss is None:
        raise RuntimeError("model must return a loss when targets are provided")
    
    # 反向传播
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)

    # 更新参数
    optimizer.step()

    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
    }