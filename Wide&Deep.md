# Wide&Deep

Wide&Deep[^1]模型混合了线性模型(Shallow)和深度神经网络(Deep)，同时具有记忆能力与泛化能力。Deep部分模型与FNN模型相似，都为Embedding层与MLP层的组合。在最后将线性模型得到的结果与MLP返回的数值进行相加，进入Sigmoid函数得到最终的预测概率。

![](./image/Wide%26Deep/Wide%26Deep.jpg)

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
本质上线性部分与深度部分共享同一个Embedding层，此处使用另一个Embedding层取得与线性回归相同的效果。

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

**深度部分MLP层**

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

**Wide&Deep模型**

```python
class WideAndDeepModel(torch.nn.Module):
    """
    A pytorch implementation of wide and deep learning.

    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))
```

[^1]:[Wide & Deep Learning for Recommender Systems, 2016](https://arxiv.org/pdf/1606.07792.pdf)