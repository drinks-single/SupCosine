from util import *
from SupCosine import SupCosine
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import datetime
import os

def main(args):


    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.infered:
        g_list,num_label,max_nodes_num = infered_load_data(args.dataset,True,False)
    else:
        g_list,num_label,max_nodes_num = load_data(args.dataset,args.degree_as_tag)
    preprosses_data(g_list,max_nodes_num,args.seed,args.infered)
    BFS(g_list,3,max_nodes_num)

    # model = SupCosine(g_list[0].node_features.shape[-1],num_label,len(g_list),max_nodes_num, args.bfs,
    #                 args.batch_size, 0.01,num_infoGraph=1,GIN_layers = args.GIN_layers,
    #                 mlp_layers = args.mlp_layers,GIN_dropout = args.final_dropout,temperature = args.supgcon_temperature,device=device)
    
    model = SupCosine(g_list[0].node_features.shape[-1],num_label,len(g_list),max_nodes_num, 3,args.batch_size, 0.01,1,5,2,0.5,0.07,device=device)

    optimizer = optim.RMSprop(model.parameters(),lr = 0.01,alpha=0.9,weight_decay = 0.001)
    accs = []

    for fold in range(args.folds):

        train,test,train_dsi,test_dsi= divide_train_test(g_list,seed = args.seed,fold_idx = fold,num_class=num_label)
        max_fold_acc = 0
        eva_acc_record=[]
        tbar = tqdm(range(args.num_epoch))

        for epoch in tbar:
            train_loss = 0
            batch_num = 0
            train_out = []

            for i in range(0,len(train),args.batch_size):
                
                model.train()
                x_batch,x_batch_adj,x_batch_bfs,x_batch_dsi,x_batch_dsi_adj,\
                    x_batch_bfs_dsi,x_labels=load_batch(train,train_dsi,i,args.batch_size,args.device)
                

                output,loss= model(x_batch,x_batch_adj,x_batch_bfs,x_labels,\
                    x_batch_dsi,x_batch_dsi_adj,x_batch_bfs_dsi,0.8,0.1)
               
                total_loss = loss

                if optimizer is not None:
                    optimizer.zero_grad()
                    total_loss.backward()         
                    optimizer.step()
                
                batch_num += 1
                train_loss += total_loss
                
                train_out.append(output.detach())

            train_out = torch.cat(train_out,0)
            pred  = train_out.max(1,keepdim = True)[1]
            labels = torch.LongTensor([graph.label for graph in train]).to(device)
            correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
            train_acc = correct/float(len(train))

            model.eval()
            test_out = []
            for i in range(0,len(test),args.batch_size):
                x_batch,x_batch_adj,x_batch_bfs,x_batch_dsi,x_batch_dsi_adj,\
                    x_batch_bfs_dsi,x_labels=load_batch(test,test_dsi,i,args.batch_size,args.device)

                output,_= model(x_batch,x_batch_adj,x_batch_bfs,x_labels,\
                    x_batch_dsi,x_batch_dsi_adj,x_batch_bfs_dsi,0.8,0.1)
                    
                test_out.append(output.detach())
            test_out = torch.cat(test_out,0)
            pred = test_out.max(1,keepdim =True)[1]
            labels = torch.LongTensor([graph.label for graph in test]).to(device)
            correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
            eva_acc = correct/float(len(test))

            #show result
            if eva_acc > max_fold_acc:
                max_fold_acc = eva_acc
            tbar.set_description_str("folds {}/{}".format(fold + 1, args.folds))
            tbar.set_postfix_str("loss: {:.2f}, best_acc:{:.2f},train_acc:{:.2f}".format(train_loss / batch_num, max_fold_acc,train_acc))

            try:
                eva_acc_record.append(eva_acc)
            except:
                eva_acc_record = [eva_acc]
        accs.append(max_fold_acc)
    accs = np.array(accs)
    mean = np.mean(accs) * 100
    std = np.std(accs) * 100
    ans = {
        "mean": mean,
        "std": std
    }
    return ans
    
if __name__ == "__main__":
    args = args_parse()

    save_path = os.path.join('result',args.dataset)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.infered:
        r_filename = str(datetime.date.today())+'-n-result.txt'
    else:
        r_filename = str(datetime.date.today())+'-result.txt'

    result_path = os.path.join(save_path,r_filename)


    ans = main(args)              
    with open(result_path,'a+') as f:
        s=args.dataset+' '
        s+=str(ans['mean'])+' '+str(ans['std'])+'\n'
        f.write(s)
    f.close()

    print(ans)