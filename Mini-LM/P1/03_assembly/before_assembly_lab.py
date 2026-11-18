"""P1 Tiny Transformer 拼装前复习版（2026-11-18 重建）。

这是依据当前接口重建的练习模板，不是最初文件的逐字历史备份。
只提供接口、配置默认值、TODO 和提示；没有填写核心实现。
学习者完成的版本保留在同目录 assembly_lab.py，本文件不导入其答案。

顺序：配置准备 → A SwiGLU → B Attention → C Block → D LM → E batch → F step。
复用已有 Tiny_transformer.py 和 RotaryEmbedding.py；不重写其内部算法。
范围：float32、dropout=0、无 padding 输入、无 KV Cache、单设备。
符号：B=batch，T=序列长度，C=模型维度，H=头数，D=C/H，V=词表大小。

NotImplementedError 表示“尚未实现”，不是输入校验；完成对应 TODO 后删除。
目前连配置检查也是练习，因此实例化 ModelConfig 会先触发占位异常。
本文件不附带测试脚本或自动训练入口，不在导入时创建模型或设置随机种子。
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from Tiny_transformer import TransformerBlock
from RotaryEmbedding import RotaryEmbedding as RoPE


# TODO 0：确认依赖模块可以安全导入。
# 演示和随机种子设置应放在各文件的 __main__ 判断内；类定义留在外面。
# 已经完成的导入保护不必重新修改。这里不要求编写冒烟测试。


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
        # TODO 0A：检查各维度为正、C 能被 H 整除、RoPE 的 D 为偶数。
        # norm_eps 与 rope_base 应为正；检查失败时给出明确错误信息。
        raise NotImplementedError("TODO 0A: validate configuration")

    @property
    def head_dim(self) -> int:
        # TODO 0B：根据模型维度和头数计算每个头的维度。
        raise NotImplementedError("TODO 0B: derive head dimension")


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # TODO A1：注册 gate_proj、up_proj、down_proj，均不使用 bias。
        # 前两者 C→hidden_dim，最后一个 hidden_dim→C。
        # 使用传入的 config 实例；不要读取 ModelConfig 类的默认值代替它。
        raise NotImplementedError("TODO A1: create SwiGLU projections")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO A2：[B,T,C] → 两条 [B,T,hidden_dim] 分支 → [B,T,C]。
        # SiLU 用于 gate 分支；与 up 分支逐元素相乘，再进行下投影。
        # 不改变 batch/序列维度，不在这里混合不同 token。
        raise NotImplementedError("TODO A2: connect SwiGLU branches")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, dropout_p: float = 0.0):
        super().__init__()
        # TODO B1：保存后续方法要访问的配置和 dropout 概率。
        # 注册 q_proj、k_proj、v_proj、out_proj：均为 C→C，无 bias。
        # 注册 self.rope；构造参数来自 head_dim/max_seq_len/rope_base。
        # 每个 self.xxx 在读取前必须初始化，尤其是 config 与 rope_base。
        # dropout 概率范围为 [0,1]；本次拼装练习默认取 0。
        raise NotImplementedError("TODO B1: create attention modules")

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # TODO B2：[B,T,C] → [B,T,H,D] → [B,H,T,D]。
        # 先拆特征轴，再交换轴；不能直接 reshape 成最终 shape 代替交换。
        # 解释 reshape 与 transpose 对元素对应关系、stride 和存储的影响。
        raise NotImplementedError("TODO B2: split heads")

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # TODO B3：[B,H,T,D] → [B,T,H,D] → [B,T,C]。
        # 合并的是同一个 token 的不同头；先恢复轴顺序，再合并 H 与 D。
        # 非连续张量可以用 reshape，或先 contiguous 再 view。
        raise NotImplementedError("TODO B3: merge heads")

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        # TODO B4：检查输入为 [B,T,C]；位置非负且不超过 RoPE 缓存范围。
        # 数据流：投影 → 拆头 → RoPE(Q,K) → SDPA → 合头 → 输出投影。
        # RoPE 不处理 V；start_pos 传给 RoPE，它不会提供历史 KV。
        # 当前 Q/K 序列等长，使用因果 SDPA，默认缩放不应重复手动执行。
        # 训练时使用配置的 dropout 概率，评估时向 SDPA 显式传入 0。
        # 返回单个 [B,T,C] Tensor，不返回 attention weights。
        # 归一化与残差已经由 TransformerBlock 负责。
        raise NotImplementedError("TODO B4: assemble attention forward")


def build_block(config: ModelConfig) -> TransformerBlock:
    # TODO C：创建新的 CausalSelfAttention 与 SwiGLU 实例。
    # 将它们和 embed_dim/norm_eps 传入已有 TransformerBlock，返回 Block。
    # 这里处理模块对象，不需要输入 x，也不执行前向计算。
    # 每次调用都创建独立子模块；可共享配置，不能意外共享各层参数。
    raise NotImplementedError("TODO C: assemble a new block")


class TinyDecoderLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # TODO D1：保存配置，注册 token_embedding、blocks、final_norm、lm_head。
        # Embedding 参数表 [V,C]；blocks 使用 ModuleList 存放 N 个独立 Block。
        # Final RMSNorm 处理最后 C 维；LM Head 为 C→V 的无 bias 线性层。
        # 每层通过新的 build_block 调用创建，不重复同一个 Block 对象。
        # Embedding 与 LM Head 共享同一个 [V,C] Parameter，不只是复制数值。
        raise NotImplementedError("TODO D1: register language model modules")

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # TODO D2：输入为 long [B,T]；batch 非空，T 在配置范围内。
        # 正常 ID 在 [0,V)；输入、标签与模型应在同一 device。
        # targets 如提供，应为同 shape 的 long，至少有一个不为 -100 的目标。
        # 数据流：Embedding → 逐层 Block → Final Norm → LM Head。
        # shape：[B,T] → [B,T,C] → [B,T,V]。
        # 每层均处理完整序列，start_pos=0；层号不是 token 位置。
        # 返回 raw logits；不额外加一次 RoPE，不提前 softmax。
        # TODO D3：无标签时返回 (logits, None)，不要访问 None 的 reshape。
        # 有标签时把 B/T 展平，计算交叉熵并返回 (logits, scalar loss)。
        # ignore_index=-100；目标已由数据端 shift，此处不得再次 shift。
        # loss 保留计算图，不提前 item/detach，不在这里 backward 或更新参数。
        raise NotImplementedError("TODO D2/D3: forward and optional loss")


def make_batch(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """输入 long [B,T+1]；返回 (input_ids, targets)，均为同 device 的 [B,T]。"""
    # TODO E：检查二维、long、非空 batch，以及每行至少两个 token。
    # 输入保留前 T 个 token，目标保留后 T 个 token，按接口顺序返回。
    # 人工核对：[1,2,3,4] 应形成输入 [1,2,3]、目标 [2,3,4]。
    # 切片可能不连续并共享原始存储；保持 dtype/device，避免原地修改。
    raise NotImplementedError("TODO E: construct next-token examples")


def train_step(
    model: TinyDecoderLM,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    # TODO F：训练模式 → 清旧梯度 → 前向得到 loss → 反向 → 裁剪 → 参数更新。
    # 模型与优化器由外部创建；这里不重新创建，也不重新移动模型 device。
    # loss 不应为 None；backward 前保持 Tensor 与计算图。
    # 使用 clip_grad_norm_，max_norm=1.0，error_if_nonfinite=True。
    # 裁剪在 backward 后、step 前；保留其返回的裁剪前总范数。
    # 返回 loss/grad_norm 两个键，值均为 Python float。
    # loss 指本次更新前的前向值；step 不会自动重新计算 loss。
    raise NotImplementedError("TODO F: perform one training update")


# 完成 A—F 后的复盘提示（不附带运行脚本）：
# 1. 固定 batch、模型和优化器放在循环外，反复调用 train_step。
# 2. 观察时使用 eval 与 no_grad；下一训练步应恢复 train 模式。
# 3. 先人工核对输入/目标，再观察 loss、逐 token 准确率与梯度范数。
# 4. 保存实验配置及真实日志；不能将简单周期序列的记忆能力当作泛化能力。
# 5. 过拟合通过也不单独证明 RoPE、因果性与所有边界条件正确。
