'''
1. 使用 V=13, C=6, 构造 [2,5] 的 input_ids。
2. 设置 padding_idx=0, 输入中至少包含两个 padding token。
3. 打印输入和输出的 shape/dtype/device/stride。
4. 验证相同 token ID 得到相同向量。
5. 使用 F.one_hot 验证查表结果与矩阵乘法一致。
6. 反向传播后找出哪些 Embedding 行有非零梯度。
7. 验证 padding 行梯度为零。
8. 计算 FP32、FP16/BF16 下 Embedding 权重占用的字节数。
9. 不照抄示例的 loss, 自己设计一个只依赖部分 token 的标量 loss。
另外回答四个问题：
1. 为什么 input_ids 必须是整数索引，而不是普通浮点特征？
2. 为什么同一个 token 在进入 Transformer 前会得到相同向量？
3. 同一 token 在一个 batch 中出现多次时，梯度会如何变化？
4. Embedding 已经包含 token 信息，为什么还需要位置编码？
'''

