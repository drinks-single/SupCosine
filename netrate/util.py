import networkx as nx
import numpy as np
import random
import os
class netGraph(object):
    def __init__(self,g,label):
        self.g = g
        self.label = label

def create_network(file):                   #这个函数写在外部，主要是net数据不一定有，可以做个比较
                                            #先处理cascade，这个函数可以进行一些纠错，比较节点数目、id之类的
    """
    File should be in txt format.
    """
    g=nx.DiGraph()
    with open('data/%s'%(file),'r') as f:   #由于netrate的文本格式和将来的数据库不一样，这里先按照netrate来写
        while True:
            line = f.readline().strip().split(",")
            if(len(line)==1):               #读取到空白行时退出，隐含了跳过这个空白行的操作
                break
            g.add_node(int(line[0]))        #创建节点数正常的图g
        
        while True:
            line = f.readline().strip().split(",")
            if len(line)!=3:
                break
            g.add_edge(int(line[0]),int(line[1]),weight=float(line[2]))
        
    f.close() 
    return g

def create_network_np(file):                   #这个函数写在外部，主要是net数据不一定有，可以做个比较
                                            #先处理cascade，这个函数可以进行一些纠错，比较节点数目、id之类的
    """
    File should be in txt format.
    """
    node_num=0
    with open('data/%s'%(file),'r') as f:   #由于netrate的文本格式和将来的数据库不一样，这里先按照netrate来写
        while True:
            line = f.readline().strip().split(",")
            if(len(line)==1):               #读取到空白行时退出，隐含了跳过这个空白行的操作
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

#废弃函数，将来可能用的上？
def create_cascades_plus(self,file):                  #直接生成每个cascade的节点序列
        """
        file should be in txt format
        """

        with open('data/%s'%(file),'r') as f:           #由于netrate的文本格式和将来的数据库不一样，这里先按照netrate来写
            while True:
                line = f.readline().strip().split(",")
                if(len(line)==1):                       #读取到空白行时退出，隐含了跳过这个空白行的操作
                    break
                self.num_nodes+=1
            

            while True:
                line = f.readline().strip().split(",")
                if len(line)<=1:
                    break
                single_cascade=[]          
                for i in range(0,len(line),2):
                    single_cascade.append([int(line[i]),float(line[i+1])])
                self.cascades.append(single_cascade)

        f.close()
        self.num_cascades=len(self.cascades)
        return self.cascades


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

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            #计算最大的节点数量
            if nodes_max<n:
                nodes_max = n
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            graph_f_t=[]#每个图的特征节点总和
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
            

    for g in g_list:        #修正图的标签
         g.label = label_dict[g.label]

    print('# classes: %d' % len(label_dict))
    #print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))
    print("# max_nodes: {}".format(nodes_max))

    return g_list, len(label_dict), nodes_feature_tag

def trans_data_to_cascade(g_list,savefile):
    data_cascades = []
    for g in g_list:
        g_cas= trans_graph_to_cascades(g.g,g.label,savefile)
        #每次加一个图的相应cascades
        data_cascades.append(g_cas)
    return data_cascades

def trans_graph_to_cascades(g,l,savefile):
    add_weight(g)
    cascades = []
    nodes_num = len(g._node)
    savefile.write(str(nodes_num)+','+str(l)+'\n')
    #生成节点序列
    for i in range(nodes_num):
        start = random.randint(0,nodes_num-1)#确定起始节点
        seq = [[start,0]]
        while True:
            #获取周围节点
            father = seq[-1]
            neibor = dict(g.adj[father[0]])
            for p in seq:
                if p[0] in neibor:
                    neibor.pop(p[0])        #删除已经出现的元素
            neibor_l = list(neibor)
            if neibor_l==[]:
                break
            #分别计算得到相应时间,这里采用相对时间
            neibor_time=[-np.log(0.4)/neibor[k]['weight'] for k in neibor]
            #计算随机量得到转移方向和时间
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
    #对节点序列中的节点赋值gamma分布
    #将相应的序列存入cascades
    return cascades

def add_weight(g):
    """
    给图本身加上符合某种分布的权重,这里暂时定为gamma函数，参数分别是1.5,0.1
    """
    # for item in g.edges:
    #     g.add_edge(item[0],item[1],weight = np.random.gamma(3,0.1,1)[0])

    #采用了新的方式赋予合适的权重，公式这里设置为按照两个节点度数的差值进行计算。
    #结果并不理想，采用别的方式进行处理
    # for node in g:
    #     neib_degree_total = 0
    #     for neighbor in g.adj[node]:
    #         neib_degree_total+=g.degree(neighbor)
    #     if neib_degree_total==0:
    #         continue
    #     for neighbor in g.adj[node]:
    #         g.add_edge(node,neighbor,weight = g.degree[neighbor]/neib_degree_total)

    for edg in g.edges:
        weight = np.abs(g.degree(edg[0])-g.degree(edg[1]))
        weight = 0.5+0.5*np.tanh(weight)
        g.add_edge(edg[0],edg[1],weight=weight)

    return 0

def generate_rand_time(l):
    """
    用于实现加权随机，返回的是选择的下标，l中存放的是各个下标的权重
    """
    w = [1/t for t in l]
    sum_w = sum(w)
    pos = random.uniform(0,sum_w)
    for i,val in enumerate(w):
        pos-=val
        if pos<=0:
            return i            
    return 0


#有些测试用的代码，先直接写在这边
    # network = np.load('network.npy')
    # predicted = np.load('cas_work.npy')
    # x = np.arange(0,10,0.1)
    # y = network.reshape(-1)
    # Y = np.ma.masked_equal(y,0)
    # c = []
    # for item in y:
    #     if item>0:
    #         c.append(item)
    # d=np.random.gamma(shape=2,scale=0.05, size = 1000)
    # plt.subplot(121)
    # plt.hist([x,c],bins=200,alpha=0.5)
    # plt.subplot(122)
    # plt.hist([x,d],bins=200,alpha=0.5)
    # plt.show()