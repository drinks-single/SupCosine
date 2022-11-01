import torch.nn as nn
import torch.nn.functional as F
import sys
import torch
import math

from torch.nn.parameter import Parameter
sys.path.append("./submodel/GIN")
from layers import GraphConvolution
from mlp import MLP

class GIN(nn.Module):

    def __init__(self, layers,num_mlp_layers,nfeat, nhid,nclass,num_nodes,dropout,device):
        super(GIN, self).__init__()

        self.device = device
        self.gcn = nn.ModuleList()
        self.gcn.append(GraphConvolution(nfeat, nhid,device = device,bias=False))
        for layer in range(layers-1):
            self.gcn.append(GraphConvolution(nhid, nhid,device = device,bias=False))

        self.dropout = dropout
        self.output = nclass
        self.layers = layers

        self.p  = Parameter(torch.empty(size=(nfeat,1))).to(device)
        nn.init.normal_(self.p,mean=1,std=0.1)
        self.dense = nn.Linear(in_features=nhid,out_features=nclass).to(device) 
        self.mlps = nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(layers-1):
            if layer==0:
                self.mlps.append(MLP(num_mlp_layers,input_dim=nfeat,hidden_dim=nhid,output_dim=nhid,device =device))
            else:
                self.mlps.append(MLP(num_mlp_layers,input_dim=nhid,hidden_dim=nhid,output_dim=nhid,device =device))
            self.batch_norms.append(nn.BatchNorm1d(num_nodes).to(device))

        self.linear_prediction = torch.nn.ModuleList()
        for layer in range(layers):
            if layer==0:
                self.linear_prediction.append(nn.Linear(nfeat,nclass).to(device))
            else:
                self.linear_prediction.append(nn.Linear(nhid,nclass).to(device))


    def _preprocess_projection_pool(self,x,adj,k):
        select_num = int(x.shape[1]*k)
        mean_sum = (torch.matmul(x,self.p)/torch.sum(torch.square(self.p))).reshape([-1,x.shape[1]])
        a_top,a_top_idx = torch.topk(mean_sum,select_num)
        a_shape = mean_sum.shape
        a_top_sm = a_top*0+1

        a_input = torch.zeros(a_shape,dtype = torch.float).to(self.device)
        result = torch.scatter(a_input,1,a_top_idx,a_top_sm)

        return result

    def forward(self, data,adj,k):

        pool = self._preprocess_projection_pool(data,adj,k)

        a_index = torch.unsqueeze(pool,-1).repeat(1,1,data.shape[-1])    
        x = a_index*data

        for layer in range(self.layers-1):
            pooled = torch.matmul(adj,x)
            pooled_rep = self.mlps[layer](pooled)
            x = self.batch_norms[layer](pooled_rep)
            x = F.relu(x)
        a_index = torch.unsqueeze(pool,-1).repeat(1,1,x.shape[-1])
        h = a_index*x

        logits = self.dense(h)
        logits = F.leaky_relu(logits)
        a_index = torch.unsqueeze(pool,-1).repeat(1,1,logits.shape[-1])
        logits = a_index*logits

        return h,logits

