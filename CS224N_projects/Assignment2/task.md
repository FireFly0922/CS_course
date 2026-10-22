# Assignment2 Text

## 1. understanding word2vec

### (a) 对$-\sum_{w \in \text{Vocab}} y_w\log(\hat y_w) = -\log(\hat y_o)$的理解

先回忆一下这个公式是从哪里来的：\
cross entropy 中进行 one-hot 操作，只有一个词是正确的，也就是只有正确位置 o 满足$y_o = 1$有非零的系数，其它系数均为0

### (b) 梯度

#### (i) 矩阵运算：求梯度公式推导

loss拆解为：$$J = -u_o^{\top}v_c + \log (\sum_w e^{u_w^T v_c}) = A+B$$
数学上$$\frac{\partial A}{\partial v_c} = -u_o^{\top}$$
一般要求梯度的 shape 要和$v_c$统一，最终写做$$\frac{\partial A}{\partial v_c} = -u_o$$
值得注意的时对B求偏导的求法：
使用链式法则求导：
$$B = \log S, S = \sum_w e^{u_w^Tu_c}$$
$$\frac{\partial B}{\partial v_c} = \frac{1}{S} \frac{\partial S}{\partial v_c} = \frac{1}{\sum_x e^{u_x^{\top} v_c}} \sum_w e^{u_w^{\top} v_c}u_w = \sum_w(\frac{e^{u_w^{\top}v_c}}{\sum_x e^{u_x^{\top}v_c}})u_w$$
最终 softmax 自然出现，有：
$$\frac{\partial A}{\partial v_c} + \frac{\partial B}{\partial v_c} = -u_o + \sum_w \hat{y}_w u_w$$
转换成矩阵形式
$$\sum_w \hat{y}_wu_w = U \hat y$$
$$-u_o = Uy$$
注意矩阵的 shape ，最终
$$\frac{\partial J}{\partial v_c} = \frac{\partial A}{\partial v_c} + \frac{\partial B}{\partial v_c} = U (\hat y - y)$$

#### (ii) 梯度等于0的时刻

直观上看 $\hat y - y = 0$ 时，梯度显然为 0 \
另外，当 $(\hat y - y)$ 是 U 的零空间时，梯度也为0 \
注：零空间 (null space) 指两个矩阵相乘后为0

#### (iii) $-u_o$ 和 $\sum_w \hat{y}_w u_w$ 两项在梯度下降更新时分别有什么作用

梯度下降时$$\theta_{t+1} \leftarrow \theta_t - \alpha \nabla_{\theta_t}J_{minibatch}(\theta_t)$$也就是用右边基于旧参数$\theta_t$算出来的新值，赋给下一时刻参数$\theta_t$($\alpha$为学习率)
$$\frac{\partial J}{\partial v_c} = \sum_w(\hat y_w - y_w)u_w$$
对正确词 $w = o$,由于$y_o = 1$梯度系数为$\hat y_o - 1 < 0$，梯度下降$-\alpha(\hat y_o - 1) = \alpha(1 - \hat y_o)$将$v_c$拉向正确词 \
而且会根据错误的程度分配不同的权重 \
对错误词$w \ne o$，同理梯度的系数为$\hat y_w$，梯度下降$-\alpha \hat y_w u_w$将$v_c$推离错误词

### (c) L2 归一化

L2 normalization : $\frac{u}{||u||_2}$，什么时候这种方法会丢掉对分类有用的信息？

对比原始 embedding得到的 u ,$\frac{u}{||u||_2}$将所有非零向量的L2范数缩放为1，但保留方向\
倘若使用这种归一化方法$u_x = \alpha u_y$两词向量在归一化后将完全一样，如果$u_x$的长度信息表示语义的强度，比如 good 强度比 great 小，抛弃长度信息就失去的强度含义。\
但是长度信息不一定有意义， 有时会刻意使用L2 归一化这种方法只考虑夹角信息，即使用$cos \theta$衡量词向量相似度

### (d) 训练样本如何修改 outside embedding

对$J = -u_o^{\top}v_c + \log (\sum_w e^{u_w^T v_c}) = A+B$ , A 和 B 分别对$u_o$求导
$$\frac{\partial A}{\partial v_o} = -v_c$$
$$\frac{\partial B}{\partial v_o} = \frac{\partial \log (\sum_x e^{u_x^{\top}v_c})}{\partial v_o} = \frac{1}{\sum_x e^{u_x^{\top}v_c}}\frac{\partial}{\partial u_o} e^{u_{o}^{\top}v_c} = \hat{y}_o v_c$$
最终$$\frac{\partial J}{\partial u_o} = (\hat y_o-1)v_c$$
对错误词 $u_w (w \ne o)$
$$\frac{\partial J}{\partial u_w} = \hat y_w v_c$$
将 1(b) 中的结果和 1(d) 中的结果放在一起对比，会发现：\
对中心词$$\frac{\partial J}{\partial v_c} = \sum_w(\hat{y}_w - y_w)u_w$$
对外部词$$\frac{\partial J}{\partial u_w} = (\hat y_w - y_w)v_c$$

### (e) $\frac{\partial J}{\partial U}$的矩阵形式

$$\frac{\partial J}{\partial U} = [\frac{\partial J}{\partial u_1}, \frac{\partial J}{\partial u_2}, …, \frac{\partial J}{\partial u_{|V|}}] = [(\hat y_1 - y_1)v_c, (\hat y_2 - y_2)v_c, …, (\hat y_{|V|} - y_{|V|})v_c]$$
注意 shape ,将上面的公式写成向量形式
$$v_c(\hat y - y)^{\top}$$

## 2. Machine Learning & Neural Network

### (a) Adam 优化器

#### (i) momentum 机制

minibatch : 实际训练时经常只选择部分样本进行训练，用于计算$\nabla J_{minibatch}$ \
但是实际训练时不同的 minibatch 会有噪声，从而使梯度抖动。\
momentum 使用$$m_{t+1} = \beta_1 m_t + (1-\beta_1)\nabla_{\theta_t}J_{minibatch}(\theta_t)$$
momentum 通过$\beta$对新一 batch 梯度更新的情况和过去的梯度进行控制，防止每一batch梯度更新时抖动过严重\
不同 batch 的梯度都是对真实整体下降方向的带噪声估计，momentum 通过累计历史方向，将上下波动削弱，从而降低噪声带来的方差，得到最终长期一致的下降趋势

#### (ii) 自适应学习率

一个模型具有很多参数，这些参数不同维度的历史梯度尺度不同，假如对1000和0.01都使用同样的学习率$\alpha$进行更新，就会导致有的参数更新非常猛，有的方向几乎不动\
$$v_{t+1} = \beta_2v_t + (1 - \beta_2)(g_t \odot g_t)$$
参数更新$$\theta_{t+1} = \theta_t - \alpha \frac{m_{t+1}}{\sqrt{v_{t+1}}}$$
每次更新的量其实是$\alpha \frac{m_{t+1}}{\sqrt{v_{t+1}}}$，注意到这一步$m_{t+1}$和$\sqrt{v_{t+1}}$尺度一样，也就是进行了一个类似归一化的操作，将大小不同的参数放缩到了同一尺度上，方便$\alpha$更新

### (b) Dropout

#### (i) $\gamma$的作用

每个 minibatch 随机关闭部分隐藏层的单元
$$\mathbf h_{drop} = \gamma \mathbf d \odot \mathbf h$$
$$E_{p_{drop}}[\mathbf h_{drop}]_i = h_i$$
我们希望在经过 dropout 后计算期望时不会因为对$\mathbf d \odot \mathbf h$求期望引起原来数值尺度的变化，需要设置$\gamma$
$$E[\gamma d_i h_i] = h_i$$
由于$h_i$是隐藏层元素，是定值\
而$d_i$是一个随机 mask，记$d_i = 0$的概率为$p_{drop}$ \
最终$$\gamma = \frac{1}{1-p_{drop}}$$

#### (ii) 为什么训练时 dropout , 评估时不 dropout ?

首先，dropout的意义是防止训练过程中模型对某些固定 hidden units / 固定路径过度依赖 , 所以随机关闭一些 unit 减轻因模型依赖固定路径产生的过拟合\
在 evaluation 阶段只测试模型训练后的真实效果，如果仍 dropout , 推理结果就会产生随机性。

## 3. Dependency Parsing

实现一个 transition-based dependency parser 每一步维护 Stack , Buffer, Dependencies 三个状态\
初始状态: stack 只有 root ; buffer 中是整句话 ; Dependencies 为空\
三个动作:

- Shift ： 负责将 buffer 最前面的词移动到 stack 顶
- Left-ARC : 如果 stack 顶部是[…, A, B], 建立$B \rightarrow A$并删除A
- Right-ARC : 如果 stack 顶部是[…, A, B], 建立$A \rightarrow B$并删除B
