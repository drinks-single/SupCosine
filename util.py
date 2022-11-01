import numpy as np
import argparse
import networkx as nx
from sklearn.model_selection import StratifiedKFold
import torch
import random
import os.path as osp

num_infoGraph = 1
def args_parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="MUTAG")
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device',type = int,default=0)
    parser.add_argument('--folds',type = int,default = 10,help='how much folds in train')
    parser.add_argument('--seed',type = int, default = 42,help='random seed setting')   
    parser.add_argument('--infered',type=bool,default=True,help='Whether use inferd graph')   

    return parser.parse_args()


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.max_neighbor = 0
        self.tensor_adj=None
        self.bfs_adj = None
        self.infered_A = None

def data_preprocess(dataset):
    adj = np.load('dataset/' + dataset + '/adj.npy', allow_pickle=True)
    feature = np.load('dataset/' + dataset +
                      '/features.npy', allow_pickle=True)
    subadj = np.load('dataset/' + dataset + '/sub_adj.npy', allow_pickle=True)
    label = np.load('dataset/' + dataset +
                    '/graphs_label.npy', allow_pickle=True)

    new_label = np.array([np.argmax(one_hot) for one_hot in label])

    label_idx = []
    for i in range(label.shape[-1]):
        tmp = np.where(new_label == i)
        label_idx.append(tmp)
    return feature, new_label, label.shape[1], np.ones([adj.shape[0], adj.shape[1]]), subadj, adj, adj.shape[1], \
           subadj.shape[-1], label_idx

def load_data(dataset, degree_as_tag=False):
    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('dataset plus/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        max_nodes_num = 0
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n
            if n>max_nodes_num:
                max_nodes_num = n
            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        adj_mx = np.zeros([len(g.neighbors),len(g.neighbors)],dtype = int)
        for i in range(len(g.g)):
            for j in g.neighbors[i]:
                adj_mx[i][j] = 1
        g.tensor_adj = torch.FloatTensor(adj_mx)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict),max_nodes_num


def infered_load_data(dataset,top_k,degree_as_tag=False):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
        add infered module
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    infered_file = osp.join('dataset plus/infered_A','zhongjian'+dataset+'.txt')
    infered_A = []

    with open(infered_file,'r') as f:
        while True:
            num=f.readline().strip()
            if num:
                num = int(num)
                g=[]
                for i in range(num):
                    line = f.readline().strip().split(',')
                    hang=[]
                    for j in line:
                        hang.append(float(j))
                    g.append(hang)
                infered_A.append(g)
            else:
                break

    
    count=0
    minus_count=0
    max_nodes_num = 0

    with open('dataset plus/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if n>max_nodes_num:
                max_nodes_num = n
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])
                infered_line = infered_A[i][j]
                infered_line = [[infered_line[i],i] for i in range(len(infered_line))]
                infered_line.sort(reverse=True)

                if top_k>len(infered_line):                         
                    continue
                for top_i in range(top_k):
                    if (j,infered_line[top_i][1]) not in g.edges and infered_line[top_i][0]>0:
                        g.add_edge(j,infered_line[top_i][1])
                        count+=1
                    if (j,infered_line[-top_i-1][1]) in g.edges and infered_line[-top_i-1][0]>0:
                        g.remove_edge(j,infered_line[-top_i-1][1])
                        minus_count+=1


            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n
            if n>max_nodes_num:
                max_nodes_num = n

            g_list.append(S2VGraph(g, l, node_tags))
            g_list[-1].infered_A = torch.FloatTensor(infered_A[i])
            g_list[-1].infered_A = (g_list[-1].infered_A!=0).float()

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        adj_mx = np.zeros([len(g.neighbors),len(g.neighbors)],dtype = int)
        for i in range(len(g.g)):
            for j in g.neighbors[i]:
                adj_mx[i][j] = 1
        g.tensor_adj = torch.FloatTensor(adj_mx)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))
    print("# maximun nodes number: {}".format(max_nodes_num))
    print("# data: %d" % len(g_list))
    print("# add infered edges: %d" %count)
    print("# delete infered edges: %d" %minus_count)

    return g_list, len(label_dict),max_nodes_num

def divide_train_test(graph_list,seed, fold_idx,num_class):

    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    graph_cls = [[]for _ in range(num_class)]

    for i,g in enumerate(graph_list):
        graph_cls[g.label].append(i)

    negative = []
    for i in range(num_infoGraph):
        for item in graph_list:
            temp_label =list(range(num_class))
            temp_label.remove(item.label)
            choose_label = random.sample(temp_label,1)[0]
            negative.append(graph_list[random.sample(graph_cls[choose_label],1)[0]])

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    train_graph_dsi = [negative[i] for i in train_idx]
    test_graph_dsi = [negative[i] for i in test_idx]

    return train_graph_list,test_graph_list,train_graph_dsi,test_graph_dsi

def load_batch(train,train_dsi,i, batch_size,device):

    data_size = len(train)

    if i+batch_size>data_size:
        batch,batch_dsi = train[i:data_size],train_dsi[i*num_infoGraph:data_size*num_infoGraph]

    else:
        batch,batch_dsi =train[i:i+batch_size],train_dsi[i*num_infoGraph:i+(batch_size)*num_infoGraph]
    label =[graph.label for graph in batch]

    x = torch.cat([graph.node_features.unsqueeze(0) for graph in batch],0).to(device)
    x_dsi = torch.cat([graph.node_features.unsqueeze(0) for graph in batch_dsi],0).to(device)
    node_num = len(train[0].node_features)

    x_adj = torch.cat([graph.tensor_adj.unsqueeze(0) for graph in batch],0).to(device)
    x_bfs_adj = torch.cat([graph.bfs_adj.unsqueeze(0) for graph in batch],0).to(device)
    
    x_dsi_adj = torch.cat([graph.tensor_adj.unsqueeze(0) for graph in batch_dsi],0).to(device)
    x_bfs_adj_dsi = torch.cat([graph.bfs_adj.unsqueeze(0) for graph in batch_dsi],0).to(device)

    return x,x_adj,x_bfs_adj,x_dsi,x_dsi_adj,x_bfs_adj_dsi,label

def preprosses_data(g_list,max,seed,infered=False):

    np.random.seed(seed)
    np.random.shuffle(g_list)
    for g in g_list:
        oldnum=g.node_features.shape[0]
        supnum = max-oldnum
        sup_tensor = torch.zeros(supnum,g.node_features.shape[1])
        g.node_features = torch.cat([g.node_features,sup_tensor])
        if infered:
            supadj = torch.zeros(supnum,oldnum)
            g.infered_A = torch.cat([g.infered_A,supadj])
            supadj = torch.zeros(max,supnum)
            g.infered_A = torch.cat([g.infered_A,supadj],axis=1)

        supadj = torch.zeros(supnum,oldnum)
        g.tensor_adj = torch.cat([g.tensor_adj,supadj])
        supadj = torch.zeros(max,supnum)
        g.tensor_adj = torch.cat([g.tensor_adj,supadj],axis=1)

def BFS(g_list,bfs_limit,nodes):

    for item in g_list:
        item.bfs_adj = torch.zeros_like(item.tensor_adj)

        a = item.tensor_adj.clone()        
        bfs_list = []
        bfs_list.append(a)
        a_s = a
        for i in range(bfs_limit-1):
            a_k = (torch.mm(a_s,a)>=1).float()
            bfs_list.append(((a_k-a_s)>=1).float())
            a_s = a_k
        for i in range(nodes):

            mark = bfs_limit+1
            for b in range(bfs_limit):
                if mark<=0:
                    break
                add_nodes = bfs_list[b][i].nonzero()
                if add_nodes.shape[0]==0:
                    break
                else:
                    for k in range(add_nodes.shape[0]):
                        if mark<=0:
                            break
                        if item.bfs_adj[i][add_nodes[k]]:
                            next
                        else:
                            item.bfs_adj[i][add_nodes[k]] = 1.0
                            mark-=1
    return 0