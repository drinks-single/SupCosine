import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim,device):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers
        self.device=device
        self.mlp_weight = torch.nn.ParameterList()
        self.bias =torch.nn.ParameterList()

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.mlp_weight.append(Parameter(torch.FloatTensor(input_dim,output_dim)).to(self.device))
            self.bias.append(Parameter(1,output_dim).to(self.device))
        else:
            #Multi-layer model
            self.linear_or_not = False
            
            self.batch_norms = torch.nn.ModuleList()
            
            self.mlp_weight.append(Parameter(torch.FloatTensor(input_dim,hidden_dim).to(self.device)))
            self.bias.append(Parameter(torch.FloatTensor(1,hidden_dim).to(self.device)))
            for layer in range(num_layers-2):
                self.mlp_weight.append(Parameter(torch.FloatTensor(hidden_dim,hidden_dim).to(self.device)))
                self.bias.append(Parameter(torch.FloatTensor(1,hidden_dim).to(self.device)))
            self.mlp_weight.append(Parameter(torch.FloatTensor(hidden_dim,output_dim).to(self.device)))
            self.bias.append(Parameter(torch.FloatTensor(1,output_dim).to(self.device)))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(num_features = 2,eps=1e-5, affine=False, track_running_stats=False))
        self.init_weight()
    
    def init_weight(self):
        for item in self.mlp_weight:
            nn.init.normal_(item,mean=1,std=0.1)
        for item in self.bias:
            nn.init.normal_(item,mean=0,std=0.1)

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layers in range(self.num_layers-1):
                h = torch.matmul(h,self.mlp_weight[layers])+self.bias[layers]
                h = self.batch_norms[layers](F.relu(h))
            h = torch.matmul(h,self.mlp_weight[-1])+self.bias[-1]
            h = F.relu(h)            
            return h