import numpy as np
import torch
import torch.nn.functional as F


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

class FieldAwareFactorizationMachine(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        # xs[i] 为field i对所有field的embedding向量
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                # field j中关于 特征 i 的embed * field i 关于 特征 j 的embed
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return ix

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

class CrossNetwork(torch.nn.Module):

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            # x*x^T*w + b = w*x * x + b
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x

class Residual_block(torch.nn.Module):
    def __init__(self, hidden_unit, dim_stack):
        super(Residual_block, self).__init__()
        self.linear1 = torch.nn.Linear(dim_stack, hidden_unit)
        self.linear2 = torch.nn.Linear(hidden_unit, dim_stack)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        orig_x = x.clone()     # Returns a copy of input
        x = self.linear1(x)
        x = self.linear2(x)
        outputs = self.relu(x + orig_x)
        return outputs

class MultipleResidualUnits(torch.nn.Module):

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
            layers.append(Residual_block(hidden_dim, input_dim))
            layers.append(torch.nn.Dropout(p=dropout))
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class InnerProductNetwork(torch.nn.Module):

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        returns: (batch_size, num_fields * (num_fields - 1) // 2)
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        return torch.sum(x[:, row] * x[:, col], dim=2)


class OuterProductNetwork(torch.nn.Module):

    def __init__(self, num_fields, embed_dim, kernel_type='mat'):
        super().__init__()
        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':
            kernel_shape = embed_dim, num_ix, embed_dim
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1
        else:
            raise ValueError('unknown kernel type: ' + kernel_type)
        self.kernel_type = kernel_type
        self.kernel = torch.nn.Parameter(torch.zeros(kernel_shape))
        torch.nn.init.xavier_uniform_(self.kernel.data)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]     # (batch_size, num_ix, embed_dim)
        if self.kernel_type == 'mat':
            # p.unsqueeze(1)  (batch_size, 1, num_ix, embed_dim)
            # p.unsqueeze(1)*self.kernel (batch_size, embed_dim, num_ix, embed_dim)
            # torch.sum(p.unsqueeze(1) * self.kernel, dim=-1) (batch_size, embed_dim, num_ix)
            kp = torch.sum(p.unsqueeze(1) * self.kernel, dim=-1).permute(0, 2, 1) # (batch_size, num_ix, embed_dim)
            return torch.sum(kp * q, -1)    #(batch_size, num_ix)
        else:
            return torch.sum(p * q * self.kernel.unsqueeze(0), -1)

class CrossNetwork(torch.nn.Module):

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            # x*x^T*w + b = w*x * x + b
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x