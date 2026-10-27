# 极简pytorch教程

before P0, something need to know

## tensor常用API

| API | 含义 |
| --- | --- |
| `x.shape` | 每个维度的大小,返回[行数,列数] |
| `x.ndim` | 维度数量 |
| `x.numel()` | 元素总数 |
| `x.dtype` | 元素类型 |
| `x.device` | CPU 或哪块 GPU |
| `x.stride()` | 每个维度移动一步要跨过多少元素, 返回(移动d0跨过的元素数,移动d1…) |
| `x.storage_offset()` | 第一个元素在 storage 中的偏移 |
| `x.is_contiguous()` | 当前布局是否连续(有一些操作会使内存不连续,eg:transpose,后面详细说) |
| `torch.tensor(data)` | 从已有数据创建 |
| `torch.zeros(shape)` | 全零 Tensor |
| `torch.ones(shape)` | 全一 Tensor |
| `torch.randn(shape)` | 标准正态分布随机值 |
| `torch.arange(n)` | 产生整数序列 |
| `torch.empty(shape)` | 分配未初始化空间 |
| `torch.manual_seed(seed)` | 固定随机种子 |

## shape 变换与内存布局

### x.view(new_shape)

在不改变数据排列的前提下，为同一块底层数据创建新解释

```python
import torch
x = torch.arange(12).reshape(3,4)
y = x.view(2,6)
print(y.shape)       # [2,6]
y[0,0] = 100
print(x)             # x[0,0]也变成了100，x和y共享底层数据
z = x.view(6,-1)     # 自动推导维度,d0=6,总元素不变,shape=[6,2]
flat = x.view(-1)    # 将tensor展平为一维,shape=[12]
```

### x.reshape(new_shape)

这里缺少一个和view对比的示例
一般不会复制底层内存

### x.permute(…)

按照给出顺序重排

```python
x = torch.randn(2, 3, 4)
y = x.permute(0, 2, 1)      # 原来的d0变成d0, d1变成d2, d2变成d1
print(y.is_contiguous())    # False
```

### x.contiguous()

返回内容相同，按指定格式连续存放的tensor\
不同于reshape, contiguous有可能发生真实的数据复制

```python
import torch

x = torch.arange(12, dtype = torch.float32).reshape(3,4)
print(x.shape)              # [3,4]
print(x.stride())           # (4,1)
print(x.is_contiguous())    # True

y = x.transpose(0,1)
print(y.shape)              # [4,3]
print(y.stride())           # (1,4),这里transpose没有真的移动数据，只是改变了解释底层内存的方法
print(y.is_contiguous())    # False

z = y.contiguous()          # 将y内存改为连续存放
print(z.shape)              # [4,3]
print(z.stride())           # (3,1)
print(z.is_contiguous())    # True
```

## 广播与矩阵乘

### 矩阵乘

要求两个矩阵的shape满足:左矩阵的最后一维，等于右矩阵的倒数第二维

```python
import torch
A = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
B = torch.tensor([[1.0,2.0],[3.0,4.0],[5.0,6.0]])
C1 = A @ B
C2 = torch.matmul(A,B)
C3 = A.matmul(B)
# 这里C123等价，只是不同的写法
```

### 逐元素乘

要求两个tensor的shape完全一样

```python
x = torch.randn(2,2)
y = torch.randn(2,2)
print(x * y)
```

### 广播

广播允许不同shape的Tensor进行逐元素的运算，不需要手动复制小的Tensor
广播通常不会真的复制进行扩展

```python
import torch
x = torch.tensor([[1,2,3],[4,5,6]])
bias = torch.tensor([10,20,30])
y = x + bias
print(y)                                  # [[11,22,33],[14,25,36]],纵向量也同理进行广播
```

注意：torch.zeros(2)是shape=[1,2]的向量,广播时需要注意

## nn.Module : 模型组成

### 基本接口

| API | 用途 |
| --- | --- |
| `model(x)` | 执行 forward |
| `model.parameters()` | 返回可训练参数 |
| `model.named_parameters()` | 同时返回名称和参数 |
| `model.state_dict()` | 获取模型状态 |
| `model.load_state_dict(...)` | 加载状态 |
| `model.to(device)` | 移动模型 |
| `model.train()` | 训练模式 |
| `model.eval()` | 评估模式 |

### nn.linear

```python
nn.Linear(in_features, out_features, bias=True)
```

```math
W * X + b
```

```math
W:[D_{out}, D_{in}]   b = [D_{out}]
```
bias表示是否增加偏置

nn.ReLU()：负数变成零\
nn.Dropout(p)：训练时随机关闭部分激活\
nn.Embedding(V, D)：将 token id 映射到长度为 D 的向量\
nn.ModuleList(...)：保存多个子模块,但不定义数据如何流动\
nn.Sequential(...)：按顺序执行多个模块

## Autograd : 自动反向转播

### 保存中间结果便于反向传播

`save_for_backward()`可以在前向时保存张量，并在`saved_tensors`在反向传播期间检索张量\
可以使用hook控制已保存的tensor的打包/解包方式，其接受一个map函数输出的tensor作为唯一参数，返回一个用于反向传播的tensor

```python
class SelfDeletingTempFile():
    def __init__(self):
        self.name = os.path.join(tmp_dir, str(uuid.uuid4()))

    def __del__(self):
        os.remove(self.name)

def pack_hook(tensor):
    temp_file = SelfDeletingTempFile()
    torch.save(tensor, temp_file.name)
    return temp_file

def unpack_hook(temp_file):
    return torch.load(temp_file.name)
```

注册已保存tensor的hook

```python
x = torch.randn(5, requires_grad=True)
y = x.pow(2)
y.grad_fn._raw_saved_self.register_hooks(pack_hook, unpack_hook)
```

### 不可微函数的梯度

对很多不可微函数，无法进行微分(如:relu,sqrt在0处)：\
每个算子预先定义反向传播规则，遇到不可微点时，按照约定选一个可用的梯度，之后继续链式法则求导\
pytorch约定，对局部凸函数(指开口向上的函数)，即使没有唯一的导数，可以令其有一组合理的斜率，称为"次梯度"，将这一组集合称为"次微分"\
尽管这些斜率都合理，但是pytorch需要选出一个，规定选择"范数最小的"\
范数：一个数/向量的大小，标量的范数是其绝对值；向量的范数为其L2范数$`||g||_2 = \sqrt{g_1^2+g2^2+…}`$\
如果有多个合法候选梯度，选择长度最短、最接近0向量的\
以$`z = max(x,y)`$为例:候选梯度为$`(a,1-a)`$,最小范数选择$`(0.5,0.5)`$

1.连续函数导数为无穷大\
$`\sqrt{0}`$的导数可能时无穷大，此时在0处有意义，pytorch会返回inf(这在训练中有可能导致梯度爆炸)\

2.函数本身没有定义(超出定义域)\
工程中经常加入一个很小的数

```python
eps = 1e-8
y = torch.sqrt(x + eps)
```

或者限制输入范围

```python
x_safe = torch.clamp(x, min=1e-8)
# clamp的用法是将x和(min,max)比较，取在[min,max]区间内的部分，超出的扔掉，而如果min/max超了，不取这两点
y = torch.log(x_safe)
```

3.除以0

```python
import torch
x = torch.tensor([1.0, 1.0], requires_grad=True)
divisor = torch.tensor([0.0, 1.0])
mask = divisor != 0
loss = (x[mask] / divisor[mask]).sum()
loss.backward()
print(x.grad)
```

需要先筛选有效元素，之后进行除法运算，再计算loss

4.离散操作\
对处处不可微的情况，训练时常使用 softmax 代替硬选择

```python
probabilities = torch.softmax(x, dim=0)
```

### requires_grad=True

1.计算图：以$`y=wx`$、$`L = (y-b)^2`$为例\
前向的过程中

```text
w ──┐
    ├─ 乘法 -→ y ──┐
x ──┘              ├─ 减法 -→ d -→ 平方 -→ loss
              7 ───┘
```

这就是一张计算图：节点表示 tensor/运算；forward是从左向右的过程；backward从右向左计算梯度\
`loss.backward`过程中pytorch沿计算图反向计算，并将参数梯度进行累积\

2.叶子 Tensor\
定义为计算图的原始输入(输入的)，不是计算产物\
可以用`grad_fn`检查生成该tensor的运算，叶子Tensor的`grad_fn = None`\
由于优化器最终只需要更新模型参数，通常是叶子tensor，所以中间tensor的梯度只在反向传播时临时使用以节省内存\
如果希望查看非叶子tensor的梯度

```python
y.retain_grad()          # y为非叶子tensor
loss.backward()
print("w.grad:", w.grad)
print("y.grad:", y.grad)
```

3.为什么forward占用显存?\
训练时，中间的非叶子tensor的梯度需要保存，一般训练显存包含:参数、梯度、optimizer state、forward 保存的激活值、临时 workspace

## Cross-Entropy Loss

需要未经过归一化的logits,非softmax后的概率\
一般使用`nn.CrossEntropyLoss(logits, targets)`\
在数学上,"softmax + 交叉熵"可以被化简为"LogSoftmax + NLLLoss",下面展示其推导，也证明了为什么交叉熵接收logits即可:\
模型输出的logits为$`Z = [z_1, z_2, …, z_C]`$\
softmax得到概率:$`p_i = \frac{e^z_i}{\sum_j e^{z_j}}`$\
记正确类为y,Cross-Entropy Loss为$`L = -\log{p_y}`$\
对应softmax得到的概率为$`L = -\log(\frac{e^{z_y}}{\sum_j e^{z_j}}) = z_y - \log{\sum_j e^{z_j}}`$\
对比NLLLoss,取正确类的位置并加负号 : $`L = -\operatorname{log\_softmax}(z)_y`$
```math
\operatorname{CrossEntropyLoss}(z,y) = \operatorname{NLLLoss}(\operatorname{log\_softmax}(z),y)
```
其实在Cross-Entropy内部已经实现了log_softmax，所以直接接收logits即可

## AdamW

```python
torch.optim.AdamW(
    model.parameters(),
    lr=...,
    betas=(...,...),
    eps=...,
    weight_decay=...
)
```

这里只展示接口，AdamW的原理在`CS224N_projects\Assignment2\task.md`的`2.(a)`中有过详细解释

## 模型训推组成

### 训练一个step

```text
1. 取一个 batch
2. 清空旧梯度
3. forward 得到 logits
4. 用 logits 和 targets 计算 loss
5. backward 计算梯度,不更新参数
6. 检查梯度或裁剪梯度
7. optimizer.step 更新参数
8. 记录 loss
```

#### Dropout

关于Dropout的定义以及细节在`CS224N_projects\Assignment2\task.md`的`2(b)`，在使用时

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
dropout = nn.Dropout(p=0.5)      # 此处p为关闭概率，而不是保留概率
x = torch.ones(10)
dropout.train()
train_output = dropout(x)
dropout.eval()                   # 推理时注意调用eval
eval_output = dropout(x)
print("input:", x)
print("train:", train_output)    # train: tensor([2., 0., 0., 2., 2., 0., 0., 2., 2., 2.])
print("eval:", eval_output)      # eval: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
```

#### BatchNorm

关于BatchNorm(批归一化),主要目标为"让中间数据分布更稳定，以帮助训练"\
对不同feature，其数值差距可能非常大，BatchNorm让其先进行标准的归一化\
对一个N维的feature,记作$`x`$
BatchNorm的过程:
```math
\mu_B = \frac{1}{N}\sum_{i=1}^N x_i
```

```math
\sigma_B^2 = \frac{1}{N}\sum_{i=1}^{N}(x_1-\mu_B)^2
```
标准化，为了防止除以0，引入$`\epsilon`$

```math
\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
```

通过$`\gamma`$和$`\beta`$训练以恢复这一维的缩放关系:

```math
y_i = \gamma \hat{x_i} + \beta
```
训练时：\
BatchNorm使用当前batch的均值和方差，在forward时自动更新running_mean(训练期间累计的均值)和running_var(训练期间累计的方差)\
评估时：\
$`\mu`$使用running_mean;$`\sigma^2`$使用running_var，以及训练好的$`\gamma`$和$`\beta`$

### 训练与推理模式

训练阶段:

```text
使用当前 batch 的 mean/variance
更新 running_mean/running_var
使用 gamma/beta 变换标准化结果
```

loss.backward() 计算伪代码：

```python
linear1.weight.grad
linear1.bias.grad
batch_norm.weight.grad    # gamma 的梯度
batch_norm.bias.grad      # beta 的梯度
linear2.weight.grad
linear2.bias.grad

optimizer.step()
```

推理阶段:

```text
不使用当前测试 batch 的统计量
不再更新 running_mean/running_var
使用训练期间积累的 running_mean/running_var
仍然使用训练得到的 gamma/beta
```

如果调用`with torch.inference_mode()`:关闭 autograd 记录，减少纯推理阶段不必要的计算和内存开销

完整推理：

```python
model.eval()                   # 改变 BatchNorm 等模块的行为

with torch.inference_mode():   # 关闭梯度计算
    output = model(input)
```

## Device与dtype

需要保证参与同一运算的tensor在相同的device上，且dtype兼容

### device

`device = cpu/cuda`指定使用设备，其决定数据在哪里运算\
常用api，基本上只看名字就知道大致起什么作用:

```python
x.device
x.to(device)
model.to(device)
torch.cuda.is_available()
torch.cuda.device_count()
```

```python
# 自动选择cpu/cuda
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x = torch.randn(2, 3)
print("before:", x.device)
x = x.to(device)
print("after:", x.device)
print(x)
```

从加速的角度看,cpu和cuda之间的数据搬运需要经过PCIe/NVLink进入gpu显存\
因此推理系统通常尽量让"模型权重/当前激活/KV Cache/sampling所需数据"长期驻留在GPU\
这里大致解释一下KV Cache是什么：
```math
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
```
模型准备生成下一个token时，需要让当前token的Q与前面所有token的K匹配,把每个token的V向量乘对应的注意力权重，全部相加得到新的上下文向量\
假设历史token为$`K_1, K_2, K_3`$\
计算Q与K的相似度打分

```math
s_i = \frac{Q \cdot K_i}{\sqrt{d}}
```

经过softmax得到概率(注意力权重)

```math
\alpha_i = \frac{e^{s_i}}{\sum_j e^{s_j}}
```
将得到的注意力权重分别乘对应的V并求和(称为聚合)\
第一次处理提示词称为prefill:输入prompt → 并行计算prompt中的所有token → 建立KV Cache
之后开始进入decode:输入新token → 计算新token的Q、K、V → 追加新的KV到Cache内 → 新Q查询历史KV → 生成下一个token\
KV Cache的shape可以理解为[batch, kv_heads, sequence_length, head_dim]

sampling所需数据指的是Transformer最后一层会输出logits，一般只有需要将结果呈现给用户时，才把token ID取回CPU

### dtype

`dtype = float32/float16/bfloat16`指定数据精度

对FP16、FP32、BF16这三种类型都是浮点数:
```math
\text{符号位} \times \text{尾数/有效数字} \times \text{指数} = (-1)^{\text{sign}}\times \text{significand}\times2^{\text{exponent}}
```

| dtype | 总位数 | 指数位 | 尾数位 | 每元素字节 | 最大有限值 | 1附近的最小间隔(精度) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 32 | 8 | 23 | 4 | 3.40e38 | 1.19e-7 |
| FP16 | 16 | 5 | 10 | 2 | 65504 | 9.77e-4 |
| BF16 | 16 | 8 | 7 | 2 | 3.39e38 | 7.81e-3 |

FP16 尾数比 BF16 多，因此在可表示范围内精度更细；BF16 与 FP32 有相同数量的指数位，因此数值范围远大于 FP16，但有效数字较少\
为什么训练中经常使用 BF16：它保留了较大的动态范围，通常比 FP16 更不容易发生数值溢出(最大有限值更大)\
一般说的"下溢"指的是数值特别小导致浮点格式无法表示

#### 显存计算:

以INT4为例，下面的$`1024^2`$为线性层权重矩阵的shape(`W.shape = [1024, 1024]`)
```math
1024^2\times0.5\text{ bytes}=0.5\text{ MiB}
```

#### "整数量化"

不是简单的将$`FP32 \rightarrow BF16`$\
而是将浮点数映射到有限的整数格点内\
简单来说，如果直接`.to(torch.int8)`会将大量小数变成0，且浪费很多"整数格点",所以需要scale

#### “混合精度”

根据算子的数值稳定性决定操作对应的精度,示例：

```python
model.eval()
x = x.to("cuda")
model = model.to("cuda")
with torch.inference_mode():
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        output = model(x)
```

| 方案 | 权重大小 | 精度风险 | 常见用途 |
| --- | ---: | ---: | --- |
| FP32 | 4 bytes | 最低 | 基准、调试 |
| FP16 | 2 bytes | 中等，范围较小 | GPU 推理、训练 |
| BF16 | 2 bytes | 中等，有效数字较少 | LLM 训练和推理 |
| FP8 | 1 byte | 更高 | 新硬件低精度计算 |
| INT8 | 1 byte | 更高 | 权重、activation、KV Cache |
| INT4 | 约 0.5 byte | 较高 | LLM weight-only 量化 |

#### 常用的量化方法

##### 1. PTQ：训练后量化(Post-Training Quantization)

训练好的 BF16/FP16 模型 → 收集权重或校准数据 → 选择量化粒度 → 计算 scale/zero point → 量化并打包 → 测试精度和性能

- 不需要重新完整训练模型
- 成本较低
- 适合快速部署
- INT8 通常比较容易保持精度
- INT4、INT3 等低比特下更容易出现精度下降

PTQ 有时需要校准数据，尤其是激活量化；某些纯权重量化只依赖权重或只需要少量样本

##### 2. QAT：量化感知训练(Quantization-Aware Training)

训练前向过程中加入 fake quantization：浮点权重 → 模拟量化为整数 → 模拟反量化为浮点数 → 使用近似权重完成前向传播(在训练过程中就模拟量化误差，让模型主动适应低精度)

例如：
```math
W \rightarrow q= \operatorname{round}(W/s) \rightarrow \hat W=sq
```

前向传播使用 $`\hat W`$，而模型实际保存和更新的通常仍然是浮点参数

问题在于 round() 的导数几乎处处为零，所以 QAT 通常使用 STE（Straight-Through Estimator）：\
前向：真的执行 round\
反向：近似把 round 当成恒等函数\
这样梯度就可以继续传播

- 通常比简单 PTQ 更容易保持低比特精度
- 需要训练或微调
- 需要训练数据和计算资源
- 实现与调参更加复杂
- 特别适合极低比特或对精度敏感的模型

| 对比 | PTQ | QAT |
| --- | --- | --- |
| 量化时间 | 训练后 | 训练中或微调时 |
| 是否需要训练 | 通常不需要 | 需要 |
| 成本 | 较低 | 较高 |
| 数据需求 | 无或少量校准数据 | 训练/微调数据 |
| 低比特精度 | 可能下降明显 | 通常更好 |
| 部署速度 | 快 | 前期准备较慢 |

## DataLoader

```python
loader = DataLoader(
    dataset,                           # 一般需要维护接口 __len__() 和 __getitem__()
    batch_size=32,                     # 每个batch处理多少个样本
    shuffle=True,                      # 每个epoch打乱样本顺序，验证集时关闭
    num_workers=4,                     # 使用几个子进程准备数据，可以调用多个进程提前准备后面的batch需要的数据，实现GPU算当前batch，CPU准备下一个batch需要的内容(太大也会导致内存占用增加等问题)
    pin_memory=True,                   # GPU异步复制数据时，适合从锁页内存读取
    drop_last=True,                    # 最后不足batch_size的部分被抛弃，比如希望使用batch size敏感的Batch Norm时
    collate_fn=collate_function,       # 自然语言句子经常长度不同，Tensor形状不同，需要自定义进行padding
)
```

LLM进行padding的示例

```python
import torch
from torch.utils.data import Dataset, DataLoader

class TokenDataset(Dataset):
    def __init__(self):
        self.sequences = [
            [11, 12, 13],
            [21, 22],
            [31, 32, 33, 34],
            [41],
        ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return torch.tensor(
            self.sequences[index],
            dtype=torch.long,
        )

def collate_tokens(samples):
    pad_token_id = 0

    batch_size = len(samples)
    max_length = max(
        sample.shape[0]
        for sample in samples
    )

    input_ids = torch.full(
        size=(batch_size, max_length),
        fill_value=pad_token_id,
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        batch_size,
        max_length,
        dtype=torch.long,
    )

    for row, sample in enumerate(samples):
        length = sample.shape[0]

        input_ids[row, :length] = sample
        attention_mask[row, :length] = 1

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
dataset = TokenDataset()

loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=False,
    collate_fn=collate_tokens,
)

batch = next(iter(loader))

print(batch["input_ids"])
print(batch["attention_mask"])
print(batch["labels"])
```

### iter()

DataLoader是一个可迭代对象,直接使用`for batch in loader:`不需要专门维护用于迭代的量\
等价于

```python
iterator = iter(loader)
batch_1 = next(iterator)
batch_2 = next(iterator)
batch_3 = next(iterator)
```

### sampler

`shuffle=True`进行了随机采样，也可以显式设置sampler

```python
from torch.utils.data import WeightedRandomSampler

sampler = WeightedRandomSampler(
    weights=torch.tensor([1.0,1.0,5.0,5.0]),
    num_samples=8,
    replacement=True,
)
```
