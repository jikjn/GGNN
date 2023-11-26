import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def transform_L(L,out_features=100,threshold = 0.005):
    # 假设 tensor 是输入的一维张量
    # dl_plus_1 是矩阵的维度，threshold 是主对角线元素的最低阈值
    L_mat = []
    for vector in L:
        # 创建一个形状为 (dl_plus_1, dl_plus_1) 的全零矩阵
        matrix = torch.zeros((out_features, out_features))

        # 获取下三角矩阵的行和列索引
        row_indices, col_indices = torch.tril_indices(out_features, out_features)

        # 使用索引填充下三角矩阵
        matrix[row_indices, col_indices] = vector

        # 调整主对角线上的元素
        diag = torch.diag(matrix)
        adjusted_diag = torch.maximum(diag, torch.tensor(threshold))
        matrix = torch.diag_embed(adjusted_diag) + torch.tril(matrix, -1)
        L_mat.append(matrix)
    L_mat = torch.stack(L_mat)
    return L_mat

class GNNLayer(Module):
    def __init__(self, in_features, out_features, dropout=0.5,for_L=False, quaternion_ff=False, act=F.relu):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quaternion_ff = quaternion_ff
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.bn = torch.nn.BatchNorm1d(out_features)
        self.for_L=for_L
        if for_L==False:
            self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        x = self.dropout(input) # Current Pytorch 1.5.0 doesn't support Dropout for sparse matrix

        x = torch.mm(x, self.weight)

        output = torch.spmm(adj, x)


        return self.act(output)

class GGNNLayer(Module):
    def __init__(self, in_features, out_features,transformation_hidden_dim=16):
        super(GGNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.L_transformation1=GNNLayer(self.in_features,transformation_hidden_dim)
        self.L_transformation2=GNNLayer(transformation_hidden_dim,(self.out_features+1)*self.out_features//2)#没有权重矩阵变不了维度
        self.mean_transformation=GNNLayer(self.in_features,self.out_features)


    def forward(self, input, adj):
        N = input.shape[0]

        standard_normal_samples = torch.randn(N, self.out_features)


        mean=self.mean_transformation(input,adj)
        L_matrices=self.L_transformation1(input,adj)
        L_matrices = self.L_transformation2(L_matrices, adj)

        L_matrices=transform_L(L_matrices,self.out_features)

        transformed_samples = torch.bmm(L_matrices, standard_normal_samples.unsqueeze(-1))

        transformed_samples = transformed_samples.view( N, self.out_features) + mean
        return transformed_samples

# adj=torch.randn(50,50)
# input=torch.randn(50,100)
# model=GGNNLayer(100,64)
# print(model(input,adj).shape)
