# 一个完整的2分类示例

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. 固定随机种子
# ============================================================

torch.manual_seed(42)


# ============================================================
# 2. 创建人工二分类数据
# ============================================================

num_samples_per_class = 128

# 类别 0：集中在 (-2, -2) 附近
class_0 = (
    torch.randn(num_samples_per_class, 2) * 0.6
    + torch.tensor([-2.0, -2.0])
)

# 类别 1：集中在 (2, 2) 附近
class_1 = (
    torch.randn(num_samples_per_class, 2) * 0.6
    + torch.tensor([2.0, 2.0])
)

features = torch.cat([class_0, class_1], dim=0)

labels = torch.cat([
    torch.zeros(num_samples_per_class, dtype=torch.long),
    torch.ones(num_samples_per_class, dtype=torch.long),
])

print("features shape:", features.shape)
print("labels shape:", labels.shape)


# ============================================================
# 3. 使用 DataLoader 产生 batch
# ============================================================

dataset = TensorDataset(features, labels)

data_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)


# ============================================================
# 4. 定义包含 BatchNorm 的模型
# ============================================================

class BatchNormClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(
            in_features=2,
            out_features=16,
        )

        self.batch_norm = nn.BatchNorm1d(
            num_features=16
        )

        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(
            in_features=16,
            out_features=2,
        )

    def forward(self, x):
        # x: [B, 2]

        x = self.linear1(x)
        # x: [B, 16]

        x = self.batch_norm(x)
        # x: [B, 16]

        x = self.relu(x)
        # x: [B, 16]

        logits = self.linear2(x)
        # logits: [B, 2]

        return logits


model = BatchNormClassifier()

print("\n模型结构：")
print(model)


# ============================================================
# 5. 查看 BatchNorm 初始状态
# ============================================================

print("\n训练前 BatchNorm 状态：")

print("gamma / weight:")
print(model.batch_norm.weight.detach())

print("beta / bias:")
print(model.batch_norm.bias.detach())

print("running_mean:")
print(model.batch_norm.running_mean)

print("running_var:")
print(model.batch_norm.running_var)


# ============================================================
# 6. 定义损失函数和优化器
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.01,
    weight_decay=0.01,
)


# ============================================================
# 7. 训练阶段
# ============================================================

model.train()

print("\n训练阶段：")
print("model.training:", model.training)
print(
    "batch_norm.training:",
    model.batch_norm.training,
)

num_epochs = 40

for epoch in range(num_epochs):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_features, batch_labels in data_loader:
        # ----------------------------------------
        # 7.1 清空上一个 batch 的梯度
        # ----------------------------------------

        optimizer.zero_grad(set_to_none=True)

        # ----------------------------------------
        # 7.2 前向传播
        # BatchNorm 使用当前 batch 的均值和方差
        # 同时更新 running_mean 和 running_var
        # ----------------------------------------

        logits = model(batch_features)

        # ----------------------------------------
        # 7.3 计算损失
        # ----------------------------------------

        loss = criterion(logits, batch_labels)

        # ----------------------------------------
        # 7.4 反向传播
        # 计算 gamma、beta 和 Linear 参数的梯度
        # ----------------------------------------

        loss.backward()

        # ----------------------------------------
        # 7.5 更新参数
        # optimizer 会更新：
        # - Linear 的 weight 和 bias
        # - BatchNorm 的 gamma 和 beta
        #
        # optimizer 不会更新：
        # - running_mean
        # - running_var
        # ----------------------------------------

        optimizer.step()

        # ----------------------------------------
        # 7.6 统计训练指标
        # ----------------------------------------

        batch_size = batch_features.shape[0]

        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=-1)

        total_correct += (
            predictions == batch_labels
        ).sum().item()

        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    if epoch == 0 or (epoch + 1) % 10 == 0:
        print(
            f"epoch={epoch + 1:02d}, "
            f"loss={average_loss:.4f}, "
            f"accuracy={accuracy:.4f}"
        )


# ============================================================
# 8. 查看训练后的 BatchNorm 状态
# ============================================================

print("\n训练后 BatchNorm 状态：")

print("gamma / weight:")
print(model.batch_norm.weight.detach())

print("beta / bias:")
print(model.batch_norm.bias.detach())

print("running_mean:")
print(model.batch_norm.running_mean)

print("running_var:")
print(model.batch_norm.running_var)

print(
    "num_batches_tracked:",
    model.batch_norm.num_batches_tracked,
)


# ============================================================
# 9. 推理阶段
# ============================================================

# eval() 会让 BatchNorm 使用 running_mean/running_var
# 而不是使用当前测试 batch 的均值和方差

model.eval()

print("\n推理阶段：")
print("model.training:", model.training)
print(
    "batch_norm.training:",
    model.batch_norm.training,
)

test_features = torch.tensor([
    [-3.0, -2.0],
    [-1.5, -1.0],
    [1.5, 2.0],
    [3.0, 2.5],
    [0.0, 0.0],
])

# inference_mode() 关闭 autograd 记录
with torch.inference_mode():
    test_logits = model(test_features)

    test_probabilities = torch.softmax(
        test_logits,
        dim=-1,
    )

    test_predictions = test_logits.argmax(
        dim=-1
    )


# ============================================================
# 10. 输出推理结果
# ============================================================

print("\ntest features:")
print(test_features)

print("\ntest logits:")
print(test_logits)

print("\ntest probabilities:")
print(test_probabilities)

print("\ntest predictions:")
print(test_predictions)