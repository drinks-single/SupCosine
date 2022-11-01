import numpy as np
import cvxpy as cp
import os.path as osp
import argparse
from util import create_network_np,judge
#用来测试的filename
#下一步工作：完成其他两个扩散方式的函数、加入进度条
#主要问题：计算t_hat累加和的约束不满足dcp原则
#FILE = "kronecker-core-periphery-n1024-h10-r0_01-0_25-1000-cascades.txt"
FILE = "test.txt"

class netrate:
    """
    Netrate graph predictor
    """

    def __init__(self,horizon,type_diffusion):
        """
        Initialization method
        :param horizon: time limitation of infection
        :param type_diffusion: The type of information diffusion
        """
        self.horizon=horizon
        self.type_diffusion=type_diffusion
        self.num_nodes=0
        self.num_cascades=0                         #cascade的数量，和produce里面num_cascade有区分
        self.cascades=[]
        self.sorted_cascades=[]
                  
    def create_cascades(self,file):                  #由于现在还没有确认netrate的计算方式，因此在这里先进行搁置
        """
        file should be in txt format
        """

        path = osp.join('data',file)

        with open(path,'r') as f:           #由于netrate的文本格式和将来的数据库不一样，这里先按照netrate来写
            while True:
                line = f.readline().strip().split(",")
                if(len(line)==1):                       #读取到空白行时退出，隐含了跳过这个空白行的操作
                    break
                self.num_nodes+=1            

            while True:
                line = f.readline().strip().split(",")
                if len(line)<=1:
                    break
                single_cascade=[float(-1)]*self.num_nodes          #创建单个时间序列对应的列表，列表每个位置值为相应节点的传染时间，-1为未传染
                sort_cascade=[]
                for i in range(0,len(line),2):
                    single_cascade[int(line[i])]=float(line[i+1])
                    sort_cascade.append([int(line[i]),float(line[i+1])])    #按照时间顺序的节点和时间顺序
                self.cascades.append(single_cascade)
                self.sorted_cascades.append(sort_cascade)
        f.close()
        self.num_cascades=len(self.cascades)
        
        return self.cascades   

    def produce(self):
        """
            cascade is a list consists of several single cascade per line.
            horizon is csb
            type_diffusion:'exp','pow' or 'ray'
        """                        
                                                                #玩意都可以写成类的feature，省点事
        #预处理需要是使用的cascades,以及相应的预定义变量
        infected_num=[0 for _ in range(self.num_nodes)]
        A_potential=np.zeros((self.num_nodes,self.num_nodes),dtype=float)
        A_bad=A_potential.copy()
        A_hat=A_potential.copy()                                #表示A的预测形状
        total_obj=0                                             #沿用了matlab版本的变量设置

        results=[]
        errors=[]
        self.pre_process_cas(infected_num)            
        self.log_survive(A_potential)                       #计算各感染节点幸存时间
        self.uninfected_survive(A_bad)                      #计算各幸存节点的累积暴露时间
        node_infected_times = [self._node_infected(j) for j in range(self.num_nodes)]
        #cvx 优化计算部分，必须得按照矩阵编写
        for i in range(self.num_nodes):
            
            if(node_infected_times[i]==0):
                continue                        #将来注意增加A_hat部分
            # 没感染的节点，断开其所有边
            a_hat=cp.Variable((self.num_nodes,))
            #t_hat=cp.Variable((node_infected_times[i],))    #注意这边cascade的形式，需要cascade生成时进行配合
            t_hat = cp.Variable((self.num_cascades,))
            #a_hat的一些约束
            #计算A_potential全零列
            all_zero_A = np.argwhere(np.all(A_potential[...,:]==0,axis=0))
            constraints = [a_hat[i[0]]==0 for i in all_zero_A]      #A_potential全零列表示相应点不可能有输入
            c_act=0
            
            for c in range(self.num_cascades):
                if self.cascades[c][i]>0:                                  #当前节点被感染过，不能等于零，不然是感染源
                    cidx=self._find_node(self.sorted_cascades[c],i)        #cidx是当前节点在当前感染序列中的下标，或说第几个感染的                    
                    temp=[]
                    for j in range(0,cidx):#主要的问题来源！                 
                        temp.append(a_hat[self.sorted_cascades[c][j][0]])
                    constraints.append(t_hat[c_act]==cp.sum(temp))         #log无法作用于矩阵，只能在这里加在约束中
                    #print(constraints[-1].is_dcp())
                    #print(constraints[-1])
                    c_act+=1        #自定义序号++
            #constraints.append(t_hat[c_act:]==0)          #为了消除警告，多加了个约束，其实没啥用
            # try:
            #     if c_act<node_infected_times[i]-1:
            #         constraints.append(t_hat[c_act:node_infected_times[i]-1]==0)
            # except:
            #     1
            #构建问题，输出结果的部分
            #obj的优化对象结果必须是单变量，不能是向量
            constraints.append(a_hat>=0)                    #限定a_hat取值范围
            obj=cp.Maximize(cp.sum(cp.multiply(-a_hat,(A_potential[:,i]+A_bad[:,i])))+cp.sum(cp.log(t_hat[:c_act])))

            prob = cp.Problem(obj,constraints)
            try:
                prob.solve(solver='SCS')#SCS优化器，木有钱买mosek，手动猫猫哭泣
                result = [prob.status,prob.value,obj.value]
                results.append(result)
                # print("status:", prob.status)
                # print("optimal value", prob.value)
                # print("optimal var",obj.value)
                total_obj+=obj.value
                A_hat[:,i]=a_hat.value
            except TypeError:
                A_hat[:i]=0
                errors.append(i)
            # except:
            #     print(errors)
            #     sys.exit(0)

        
        return total_obj,A_hat

    def _node_infected(self,k):
        count=0
        for i in range(self.num_cascades):
            count+= (1 if self.cascades[i][k]>=0 else 0)
        return count

    def _find_node(self,c,target):
        """
        used for find target in c, c is a two_dismensional array
        return the first index of c or False
        """
        for i in range(len(c)):
            if c[i][0]==target:
                return i
        return False

    def pre_process_cas(self,inf):
        for i in range(self.num_cascades):
            for j in range(1,len(self.sorted_cascades[i])):
                inf[self.sorted_cascades[i][j][0]]+=1
        return inf

    def log_survive(self,A):
        """
        calculate the log survive time of each infected node
        according to the type of diffusion chosed
        """
        if self.type_diffusion=='exp':
            self.exp_survive(A)
        elif self.type_diffusion=='pow':
            self.pow_survive(A)         #下面两个函数将来再写
        else:
            self.ray_survive(A)
    
    def exp_survive(self,A):
        for c in self.sorted_cascades:
            for i in range(1,len(c)):
                for j in range(i-1):
                    A[c[j][0]][c[i][0]]+=(c[i][1]-c[j][1])
        return 0


    def uninfected_survive(self,A):
        """
        calculate the log survive time of each uninfected node
        according to the type of diffusion chosed
        """
        if self.type_diffusion=='exp':
            self.exp_unif_survive(A)
        elif self.type_diffusion=='pow':
            self.pow_unif_survive(A)
        else:
            self.ray_unif_survive(A)
    
    def exp_unif_survive(self,A):
        for c in self.sorted_cascades:
            temp=np.asarray(c)
            for j in range(self.num_nodes):
                if j not in temp[:,0]:
                    for i in range(len(c)):
                        A[c[i][0]][j]+=(self.horizon-c[i][1])
        return 0

def main():
    """
    测试netrate运行情况的示例代码
    """
    parser = argparse.ArgumentParser(description="example for netrate python version")
    parser.add_argument("--network",type=str,default="kronecker-core-periphery-n1024-h10-r0_01-0_25-network.txt",
                        help="network file path, default is None")
    parser.add_argument("--cascade",type=str,default="kronecker-core-periphery-n1024-h10-r0_01-0_25-1000-cascades.txt",
                        help="cascade file path,default is ")#如果运行，需要修改
    parser.add_argument("--horizon",type=int,default=10,
                        help="The deadline of infection, also could be the latest infect time")
    parser.add_argument("--type_diffusion",type=str,default='exp',
                        help="chose the type of diffusion, could be exp, pow or ray",
                        choices=['exp','pow','ray'])
    args = parser.parse_args()

    #下面是正式的代码程序部分
    nr = netrate(args.horizon,args.type_diffusion)
    print("Reading network...")                     #从network文件生成相应的矩阵
    if(args.network!=""):
        network = create_network_np(args.network)
    else:
        network = 0
    print("Reading cascades...")                    #从cascade文件生成相应的矩阵
    nr.create_cascades(args.cascade)
    np.save('network.npy',network)
    total_obj,predicted = nr.produce()
    print("Show me what you've got!")
    np.save('cas_work.npy',predicted)

    judge(network,predicted,args.min_tol)

if __name__=="__main__":
    main()


           