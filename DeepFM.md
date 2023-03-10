# DeepFM 学习笔记

DeepFM模型结合了FM模型与DNN模型，可以看作是FM与PNN的组合，也可以看作是Wide&Deep模型结合FM升级版。

![](./image/DeepFM/DeepFM.jpg)

DeepFM可以分为FM部分和Deep部分，是一种典型的并行结构，他们共享特征的Embedding向量。

**FM部分**
linear + FM

![](./image/DeepFM/FM_part.jpg)

**Deep部分**
MLP

![](./image/DeepFM/Deep_part.jpg)


## Pytorch 实现

**Embedding层**

```python
class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.compat.long)
        # 服从均匀分布的初始化器，input必须为 tensor.float64
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)
```

**线性部分**

```python
class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        """_summary_
            equals to (embedding layer + linear layer)
        Args:
            field_dims (list): example: [3, 5, 6] means the dim of each feature field
        """
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        # 每一个feature由dim产生的位移，比如 [3, 5, 6] 得到offset [0, 3, 8]
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.compat.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # 使得每一个feature 的index依次递增，避免有重复的index进入embedding
        #x = x + torch.unsqueeze(x.new_tensor(self.offsets), 0)
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias
```

**FM层**
```python
class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        # dim = 1，为特征向量与特征向量对应位置做运算
        square_of_sum = torch.sum(x, dim=1) ** 2 # 返回 batch * embed_dim
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            # 如果不加 keepdim 会返回一个 size = batch 的一维向量
            # 加了 keepdim = True，返回一个 batch * 1 的2维向量
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix
```

**Deep部分MLP层**

```python
class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, hidden_dims, dropout, output_layer=True):
        """_summary_
        Args:
            input_dim (int): 输入层
            hidden_dims (list): 隐藏层
            dropout (float): dropout 的概率 p
            output_layer (bool, optional): 输出层是否维度为1. Defaults to True.
        """
        super().__init__()
        layers = list()
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = hidden_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)
```

**DeepFM模型**

```python
class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """
    
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        # 线性部分
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
    
    
    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        # 去掉维度 1
        return torch.sigmoid(x.squeeze(1))
```

[^2]:[深度推荐模型之DeepFM](https://zhuanlan.zhihu.com/p/57873613)