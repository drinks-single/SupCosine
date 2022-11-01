from torch.nn.modules.module import Module
import torch
from submodel.GCN.models import subGCN
from supgcon_loss import SupConLoss
from submodel.GIN.model import GIN

class SupCosine(Module):
    def __init__(self,embedding, ncluster, num_subg, subg_size,bfs,batch_size, learning_rate,num_infoGraph,
                GIN_layers,mlp_layers,GIN_dropout,temperature,device):
        super(SupCosine,self).__init__()
        self.ncluster = ncluster
        self.embedding = embedding
        self.num_subg = num_subg
        self.subg_size = subg_size
        self.bfs = bfs
        self.batch_size = batch_size
        self.output_dim = 32
        self.GIN_dim = 16
        self.SAGE_dim = 32
        self.sage_k = 1
        self.lr = learning_rate
        self.num_infoGraph = num_infoGraph
        self.device = device
        self.supgcon_loss = SupConLoss(temperature = temperature)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.bn = torch.nn.BatchNorm1d(num_features = 2,eps=1e-5, affine=False, track_running_stats=False)

        self.sub_model = subGCN(self.embedding,self.output_dim,self.subg_size,self.bfs+1,dropout=0.5,device = self.device)
        self.HGNN = GIN(layers = GIN_layers,num_mlp_layers=mlp_layers,nfeat = self.output_dim,nhid=16,nclass=self.ncluster,
                                   num_nodes = self.subg_size,dropout=GIN_dropout,device=self.device)
    
    def l2_norm(self,input,axis = 1):
        norm = torch.norm(input,2,axis,True)
        output = torch.div(input,norm)
        return output

    def forward(self,train,train_adj,train_bfs,train_label,train_dsi,train_dsi_adj,train_dsi_bfs,k,MI_loss):
        
        h = self.sub_model(train,train_bfs)
        h_dsi = self.sub_model(train_dsi,train_dsi_bfs)

        h_hgnn,h_hgnn_t = self.HGNN(h,train_adj,k)
        h_hgnn_dsi,d = self.HGNN(h_dsi,train_dsi_adj,k)

        vote_layer=torch.sum(h_hgnn_t,axis=1)
        vote_layer = torch.sigmoid(vote_layer)

        pos_feature = torch.sum(h_hgnn,axis = 1)
        pos_feature = pos_feature.unsqueeze(1)
        neg_feature = torch.sum(h_hgnn_dsi,axis = 1)
        neg_feature = neg_feature.unsqueeze(1)

        sc_supgcon = torch.cat([pos_feature,neg_feature],axis = 1)
        sc_supgcon = torch.sigmoid(sc_supgcon)
        sc_supgcon = self.l2_norm(sc_supgcon,axis =-1)
        train_label = torch.LongTensor(train_label).to(self.device)

        loss = self.criterion(vote_layer,train_label)+MI_loss*self.supgcon_loss(sc_supgcon,labels = train_label)
        
        return vote_layer,loss,


