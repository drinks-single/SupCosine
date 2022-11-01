import os.path as osp
import argparse
import sys
sys.path.append("./netrate")
from util import trans_data_to_cascade,load_data

def generate_simulate_data(graphdata):
    """
    生成模拟数据的主程序部分
    """
    #没什么好办法，把节点特征和相应的标签单独存放一个文件吧
    a, b, c = load_data(graphdata , False)
    # label_list = np.array([item.label for item in a], dtype=int)
    # np.savetxt('label_of_' + graphdata + '.txt', label_list, fmt='%d')
    path = osp.join('netratedata/cascades','cas_of_' + graphdata + '.txt')
    with open(path, 'a') as f:
        f.write(str(len(a))+'\n')
        trans_data_to_cascade(a, f)
    f.close()
    #生成相应的节点特征文件
    label_file = osp.join('netratedata/cascades','tag_feature_'+graphdata+'.txt')
    with open(label_file,'a') as f:
        for item in c:
            f.write(str(len(item))+'\n')
            for node in item:
                for k in range(len(node)-1):
                    f.write(str(node[k])+',')
                f.write(str(node[-1])+'\n')
    f.close()
    print("transformation has been done.")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate cascade data from dataset.")
    parser.add_argument("--dataset",type=str,default="PROTEINS",
                        help="choose the dataset for transformation")
    
    generate_simulate_data(parser.parse_args().dataset)

    