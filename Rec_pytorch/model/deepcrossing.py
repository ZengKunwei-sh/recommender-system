import torch

from model.layer import FeaturesEmbedding, MultipleResidualUnits

class DeepCrossingModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultipleResidualUnits(self.embed_output_dim, mlp_dims, dropout)
    
    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.mlp(embed_x.view(-1, self.embed_output_dim))
        # 如果没有squeeze(1)，x的shape为 batch*1，是一个二维的tensor，sigmoid函数无法处理
        #经过x.squeeze(1)后，x的shape变为 batch, 是一个一维的tensor，那么sigmoid也会返回一维tensor
        return torch.sigmoid(x.squeeze(1))