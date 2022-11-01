from tqdm import tqdm
import os.path as osp
import argparse
import sys
sys.path.append("./netrate")
from util import load_data
from netrate_code import netrate
"""
用来生成相应的netrate矩阵
"""

def load_simu_data(dataset,horizon=10,diffusion='exp'):
    """
    这个函数用于提取相应的数据，并不生成相应的图
    """
    graphs,labels=[],[]
    #g_n_feature 存放图中节点的相应标签和特征
    path = osp.join('netratedata/cascades','cas_of_'+dataset+'.txt')
    with open(path,'r') as f:
        num_of_g = int(f.readline())
        for i in range(num_of_g):
            g = netrate(horizon,diffusion)
            firstline= f.readline().strip().split(",")
            g.num_nodes, l=int(firstline[0]),int(firstline[1])
            g.num_cascades= g.num_nodes         #由于生成文件时，cascade数量和节点数量一致，因此在这里直接简化
            for k in range(g.num_nodes):
                single_cascade = [float(-1)]*g.num_nodes
                sort_cascade = []
                line = f.readline().strip().split(",")
                for j in range(0, len(line), 2):
                    single_cascade[int(line[j])] = float(line[j + 1])
                    sort_cascade.append([int(line[j]), float(line[j + 1])])
                g.cascades.append(single_cascade)
                g.sorted_cascades.append(sort_cascade)
            graphs.append(g)
            labels.append(l)
    return graphs,labels

def load_features(data,num_g):
    """
    load the feature file according to specific dataset, eg."MUTAG"
    """
    g_f_t=[]
    path = osp.join('netratedata/cascades','tag_feature_'+data+'.txt')
    with open(path,'r') as f:
        for i in range(num_g):
            #c = f.readline().strip()
            num_nodes=int(f.readline().strip())
            g = []
            for j in range(num_nodes):
                line = f.readline().strip().split(",")
                line = [float(item) for item in line]
                g.append(line)
            g_f_t.append(g)            
    return g_f_t

def generate_netrate(args):
    print("Loading data...")
    #下面的函数考虑添加节点特征
    graphs_raw,label = load_simu_data(args.dataset,args.horizon,args.type_diffusion)
    graphs_data,_,_ = load_data(args.dataset)
    graph_tag_feature = load_features(args.dataset,len(graphs_raw)) #额外的读取节点特征以及标签的代码
    
    graphs = []
    for item in tqdm(graphs_raw):
        _,g=item.produce()
        graphs.append(g)
    #保存计算得到的新的矩阵A
    file_path = osp.join("netratedata",'zhongjian'+args.dataset+'.txt')
    with open(file_path,'w+') as f:
        for matrix in graphs:
            f.write(str(len(matrix))+'\n')#每个矩阵的节点数
            for line in matrix:
                for item in line[:-1]:
                    f.write(str(item)+',')
                f.write(str(line[-1])+'\n')
    f.close()

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="generate netrated matrix according to cascades.")
    parser.add_argument("--dataset",type=str,default="PROTEINS",
                        help="choose the dataset for transformation")
    parser.add_argument("--horizon",type=int,default=10,
                        help="The deadline of infection, also could be the latest infect time")
    parser.add_argument("--type_diffusion",type=str,default='exp',
                        help="chose the type of diffusion, could be exp, pow or ray",
                        choices=['exp','pow','ray'])
    args = parser.parse_args()
    generate_netrate(args)