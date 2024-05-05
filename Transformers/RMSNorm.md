https://blog.csdn.net/qq_39970492/article/details/131125752

|          | Layer Normalization (LayerNorm)                              | Root Mean Square Layer Normalization (RMSNorm)               |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 原理     | 对特征张量按照某一维度或某几个维度进行0均值，1方差的归一化 操作<br/>LayerNorm 是一种标准化方法，它计算一个样本的均值和方差，然后使用这些来对样本进行归一化。这种方法是独立于批量大小的，使得模型更加稳定。 | RMSNorm是对LayerNorm的一个改进，**没有做re-center操作（移除了其中的均值项）**，*可以看作LayerNorm在均值为0时的一个特例*。论文通过实验证明，re-center操作不重要。<br/>RMSNorm 也是一种标准化方法，但与 LayerNorm 不同，它不是使用整个样本的均值和方差，而是使用平方根的均值来归一化，这样做可以降低噪声的影响。 |
| 公式     | $$y = \frac {x - E[x]} {\sqrt {Var[x] + \epsilon}} \times \gamma + \beta,\quad where \; Var[x] = E[(x - E[x])^2]$$ | $$y = \frac {x} {\sqrt {E[x^2]} +  \epsilon} \times \gamma$$ |
| 公式解释 | 这里的x可以理解为 张量中具体某一维度的所有元素，比如对于 shape 为 (2,2,4) 的张量 input，若指定归一化的操作为第三个维度，则会对第三个维度中的四个张量（2,2,1），各进行上述的一次计算 | 作者认为这种模式在简化了Layer Norm的同时，可以在各个模型上*减少约 7%∼64% 的计算时间* |



