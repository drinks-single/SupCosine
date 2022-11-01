from structure_infer_module.generate_cas import generate_simulate_data
from structure_infer_module.generate_infer import generate_infer
import os 
import shutil
import argparse
def arg_parser():
    parser = argparse.ArgumentParser(description="data preprocess")
    parser.add_argument("--generate_cas",type = str,default = 'True',
                        help = "whether generate new cascades ")
    parser.add_argument("--dataset",type = str,default = 'MUTAG')
    parser.add_argument("--generate_infer",type = str,default = 'True',
                        help = "whether generate new inferrence")
    parser.add_argument("--horizon",type=int,default=10,
                        help="The deadline of infection, also could be the latest infect time")
    parser.add_argument("--type_diffusion",type=str,default='exp',
                        help="chose the type of diffusion, could be exp, pow or ray")
    return parser.parse_args()          

if __name__ =='__main__':
    args = arg_parser()
    if args.generate_cas!='False':
        generate_simulate_data(args.dataset)
        generate_infer(args)
    elif args.generate_infer!='False':
        generate_infer(args)
