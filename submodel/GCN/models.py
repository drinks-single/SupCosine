import torch.nn as nn
import torch.nn.functional as F
import sys
import torch
import math

from torch.nn.parameter import Parameter
sys.path.append("./submodel/GCN")
from layers import GraphConvolution


class subGCN(nn.Module):

    def __init__(self, nfeat, nhid, n,bfs_limit, dropout,device):
        super(subGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid,device = device,bias=True,)
        self.dropout = dropout
        self.bfs_agu = float(n)/bfs_limit
        self.weight = Parameter(torch.FloatTensor(n,n)).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = self.bfs_agu/ math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, data,adj):
        
        x = F.dropout(data, self.dropout, training=self.training)
        adj = adj*self.weight        
        x = F.relu(self.gc1(x, adj))       
        return x
