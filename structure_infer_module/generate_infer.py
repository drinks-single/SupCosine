from tqdm import tqdm
import os.path as osp
import argparse
import sys
sys.path.append("./structure_infer_module")
from util import load_data
from struct_infer_code import struct_infer

def load_simu_data(dataset,horizon=10,diffusion='exp'):

    graphs,labels=[],[]

    path = osp.join('./structure_infer_module/cascades','cas_of_'+dataset+'.txt')
    with open(path,'r') as f:
        num_of_g = int(f.readline())
        for i in range(num_of_g):
            g = struct_infer(horizon,diffusion)
            firstline= f.readline().strip().split(",")
            g.num_nodes, l=int(firstline[0]),int(firstline[1])
            g.num_cascades= g.num_nodes         
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
    path = osp.join('./structure_infer_module/cascades','tag_feature_'+data+'.txt')
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

def generate_infer(args):
    print("Loading data...")

    graphs_raw,label = load_simu_data(args.dataset,args.horizon,args.type_diffusion)
    graphs_data,_,_ = load_data(args.dataset)
    graph_tag_feature = load_features(args.dataset,len(graphs_raw))
    
    graphs = []
    for item in tqdm(graphs_raw):
        _,g=item.produce()
        graphs.append(g)
    file_path = osp.join("dataset plus/infered_A",'zhongjian'+args.dataset+'.txt')
    with open(file_path,'w+') as f:
        for matrix in graphs:
            f.write(str(len(matrix))+'\n')
            for line in matrix:
                for item in line[:-1]:
                    f.write(str(item)+',')
                f.write(str(line[-1])+'\n')
    f.close()
