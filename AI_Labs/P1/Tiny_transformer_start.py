import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Pre-Norm Decoder-only Transformer Block。

    参数：
        embed_dim：隐藏维度 C
        attention：已实例化的因果自注意力模块，内部包含 RoPE
        ffn：已实例化的 SwiGLU 模块

    输入：
        x：[B, T, C]，浮点 Tensor
        start_pos：当前输入序列的起始位置，第一版固定为 0

    输出：
        output：[B, T, C]
        dtype 和 device 应与输入保持一致
    """

    def __init__(
        self,
        embed_dim: int,
        attention: nn.Module,
        ffn: nn.Module,
        norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # TODO 1：
        # 将传入的 attention 和 ffn 注册为实例属性。
        # 建议属性名：self.attention、self.ffn

        # TODO 2：
        # 创建两个独立的 RMSNorm，分别用于 Attention 和 FFN 分支。
        # normalized_shape 应是什么？eps 使用 norm_eps。
        # 建议属性名：self.attention_norm、self.ffn_norm

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        # x：[B, T, C]
        if x.ndim != 3:
            raise ValueError("x must have shape [B, T, C]")

        if x.size(-1) != self.embed_dim:
            raise ValueError("x.shape[-1] must equal embed_dim")

        # TODO 3：完成 Attention 残差分支。
        #
        # 需要组织的操作：
        # - 保留进入该分支时的 residual：[B,T,C]
        # - 使用对应的 RMSNorm，shape 不变
        # - 调用 Attention，并传入 start_pos
        # - 将分支输出与正确的 residual 相加
        #
        # 分支完成后的 x：[B,T,C]

        # TODO 4：完成 FFN 残差分支。
        #
        # 需要组织的操作：
        # - 确定这一分支应保留哪个张量作为 residual
        # - 使用对应的 RMSNorm
        # - 调用 FFN；其内部可以扩展到 [B,T,F]，
        #   但返回值必须恢复为 [B,T,C]
        # - 完成残差相加
        #
        # 分支完成后的 x：[B,T,C]

        # TODO 5：返回最终 Tensor。
        # 完成后删除下面的占位异常。
        raise NotImplementedError("Complete TransformerBlock.forward")