import os.path as osp
import argparse
import sys
sys.path.append("./structure_infer_module")
from structure_infer_module.util import trans_data_to_cascade,load_data

def generate_simulate_data(graphdata):

    a, b, c = load_data(graphdata , False)

    path = osp.join('./structure_infer_module/cascades','cas_of_' + graphdata + '.txt')
    with open(path, 'a') as f:
        f.write(str(len(a))+'\n')
        trans_data_to_cascade(a, f)
    f.close()

    label_file = osp.join('./structure_infer_module/cascades','tag_feature_'+graphdata+'.txt')
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


    