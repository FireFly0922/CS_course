import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        base: float = 10000.0,
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even"
            )

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # 每一对维度的频率：[D/2]
        inv_freq = 1.0 / (
            base ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32,
                ) / head_dim
            )
        )

        # 所有位置：[max_seq_len]
        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )

        # [max_seq_len, D/2]
        angles = torch.outer(
            positions,
            inv_freq,
        )

        # 不是可训练参数，但需要跟随模型移动 device
        self.register_buffer(
            "cos_cache",
            torch.cos(angles),
        )
        self.register_buffer(
            "sin_cache",
            torch.sin(angles),
        )

    def apply_rotation(
        self,
        x: torch.Tensor,
        start_pos: int,
    ):
        # x: [B, H, T, D]

        B, H, T, D = x.shape

        if D != self.head_dim:
            raise ValueError(
                "input head_dim does not match RoPE"
            )

        end_pos = start_pos + T

        if end_pos > self.max_seq_len:
            raise ValueError(
                "position exceeds max_seq_len"
            )

        # [T, D/2]
        cos = self.cos_cache[start_pos:end_pos]
        sin = self.sin_cache[start_pos:end_pos]

        # [T,D/2] → [1,1,T,D/2]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        # 三角函数通常使用 float32 计算更稳定
        x_float = x.float()

        # [B,H,T,D/2]
        x_even = x_float[..., 0::2]
        x_odd = x_float[..., 1::2]

        rotated_even = (
            x_even * cos
            - x_odd * sin
        )

        rotated_odd = (
            x_even * sin
            + x_odd * cos
        )

        # [B,H,T,D/2,2]
        rotated = torch.stack(
            [rotated_even, rotated_odd],
            dim=-1,
        )

        # [B,H,T,D]
        rotated = rotated.flatten(-2)

        return rotated.to(dtype=x.dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        start_pos: int = 0,
    ):
        q = self.apply_rotation(q, start_pos)
        k = self.apply_rotation(k, start_pos)

        return q, k

torch.manual_seed(42)

B = 2
H = 2
T = 4
D = 8

q = torch.randn(B, H, T, D)
k = torch.randn(B, H, T, D)

rope = RotaryEmbedding(
    head_dim=D,
    max_seq_len=32,
)

q_rotated, k_rotated = rope(
    q,
    k,
    start_pos=0,
)

print("q:", q.shape)
print("q_rotated:", q_rotated.shape)

print("k:", k.shape)
print("k_rotated:", k_rotated.shape)