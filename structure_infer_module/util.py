import networkx as nx
import numpy as np
import random

class netGraph(object):
    def __init__(self,g,label):
        self.g = g
        self.label = label

def create_network(file):                   
    """
    File should be in txt format.
    """
    g=nx.DiGraph()
    with open('data/%s'%(file),'r') as f:   
        while True:
            line = f.readline().strip().split(",")
            if(len(line)==1):               
                break
            g.add_node(int(line[0]))        
        
        while True:
            line = f.readline().strip().split(",")
            if len(line)!=3:
                break
            g.add_edge(int(line[0]),int(line[1]),weight=float(line[2]))
        
    f.close() 
    return g

def create_network_np(file):                   
                                            
    """
    File should be in txt format.
    """
    node_num=0
    with open('data/%s'%(file),'r') as f:   
        while True:
            line = f.readline().strip().split(",")
            if(len(line)==1):               
                break
            node_num+=1
        g = np.zeros((node_num,node_num))
        while True:
            line = f.readline().strip().split(",")
            if len(line)!=3:
                break
            g[int(line[0]),int(line[1])] = float(line[2])
        
    f.close() 
    return g

def net_mae (a,b):
    sum_matrix = 0
    count=0
    for i in range(len(a)):
        for j in range(len(a[i])):
            if (a[i][j]*b[i][j])!=0:
                sum_matrix+=abs((a[i][j]-b[i][j])/a[i][j])
                count+=1
    return sum_matrix/count

def net_recall(a,b,th):
    count_1=0
    count_2=0
    for i in range(len(a)):
        for j in range(len(a[i])):
            if b[i,j]>th:
                count_2+=1
                if a[i,j]>th:
                    count_1+=1
    return count_1/count_2

def net_precision(a,b,th):
    count_1=0
    count_2=0
    for i in range(len(a)):
        for j in range(len(a[i])):
            if b[i,j]>th:
                count_2+=1
            if a[i,j]>th:
                count_1+=1
    return count_2/count_1

def judge(network,predicted,tol):

    pr = [0,0]
    mae=net_mae(predicted,network)
    pr[0]=net_recall(predicted,network,tol)
    pr[1]=net_precision(predicted,network,tol)
    print(mae)
    print(pr)
    return 0


def load_data(dataset, degree_as_tag=False):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}
    nodes_feature_tag =[]
    nodes_max = 0

    with open('dataset plus/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped

            if nodes_max<n:
                nodes_max = n
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            graph_f_t=[]
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
                
                if attr:
                    node_f_t=[feat_dict[row[0]],attr]
                else:
                    node_f_t = [feat_dict[row[0]]]
                graph_f_t.append(node_f_t)

            
            nodes_feature_tag.append(graph_f_t)

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n
            g_list.append(netGraph(g, l))
            

    for g in g_list:        
         g.label = label_dict[g.label]

    print('# classes: %d' % len(label_dict))

    print("# data: %d" % len(g_list))
    print("# max_nodes: {}".format(nodes_max))

    return g_list, len(label_dict), nodes_feature_tag

def trans_data_to_cascade(g_list,savefile):
    data_cascades = []
    for g in g_list:
        g_cas= trans_graph_to_cascades(g.g,g.label,savefile)
        data_cascades.append(g_cas)
    return data_cascades

def trans_graph_to_cascades(g,l,savefile):
    add_weight(g)
    cascades = []
    nodes_num = len(g._node)
    savefile.write(str(nodes_num)+','+str(l)+'\n')

    for i in range(nodes_num):
        start = random.randint(0,nodes_num-1)
        seq = [[start,0]]
        while True:

            father = seq[-1]
            neibor = dict(g.adj[father[0]])
            for p in seq:
                if p[0] in neibor:
                    neibor.pop(p[0])        
            neibor_l = list(neibor)
            if neibor_l==[]:
                break

            neibor_time=[-np.log(0.4)/neibor[k]['weight'] for k in neibor]

            next_index = generate_rand_time(neibor_time)
            next = [neibor_l[next_index],neibor_time[next_index]+father[1]]
            if next[1]>10:
                break
            else:
                seq.append(next)

        cascades.append(seq)
        for instance in seq[:-1]:
            savefile.write(str(instance[0])+','+str(instance[1])+',')
        savefile.write(str(seq[-1][0]) + ',' + str(seq[-1][1]))
        savefile.write('\n')

    return cascades

def add_weight(g):

    for edg in g.edges:
        weight = np.abs(g.degree(edg[0])-g.degree(edg[1]))
        weight = 0.5+0.5*np.tanh(weight)
        g.add_edge(edg[0],edg[1],weight=weight)

    return 0

def generate_rand_time(l):

    w = [1/t for t in l]
    sum_w = sum(w)
    pos = random.uniform(0,sum_w)
    for i,val in enumerate(w):
        pos-=val
        if pos<=0:
            return i            
    return 0
