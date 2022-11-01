import numpy as np
import cvxpy as cp
import os.path as osp
import argparse
from structure_infer_module.util import create_network_np,judge

class struct_infer:

    def __init__(self,horizon,type_diffusion):
        """
        Initialization method
        :param horizon: time limitation of infection
        :param type_diffusion: The type of information diffusion
        """
        self.horizon=horizon
        self.type_diffusion=type_diffusion
        self.num_nodes=0
        self.num_cascades=0                         
        self.cascades=[]
        self.sorted_cascades=[]
                  
    def create_cascades(self,file):                
        """
        file should be in txt format
        """

        path = osp.join('data',file)

        with open(path,'r') as f:           
            while True:
                line = f.readline().strip().split(",")
                if(len(line)==1):                      
                    break
                self.num_nodes+=1            

            while True:
                line = f.readline().strip().split(",")
                if len(line)<=1:
                    break
                single_cascade=[float(-1)]*self.num_nodes          
                sort_cascade=[]
                for i in range(0,len(line),2):
                    single_cascade[int(line[i])]=float(line[i+1])
                    sort_cascade.append([int(line[i]),float(line[i+1])])    
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

        infected_num=[0 for _ in range(self.num_nodes)]
        A_potential=np.zeros((self.num_nodes,self.num_nodes),dtype=float)
        A_bad=A_potential.copy()
        A_hat=A_potential.copy()                                
        total_obj=0                                            

        results=[]
        errors=[]
        self.pre_process_cas(infected_num)            
        self.log_survive(A_potential)                       
        self.uninfected_survive(A_bad)                      
        node_infected_times = [self._node_infected(j) for j in range(self.num_nodes)]
        for i in range(self.num_nodes):
            
            if(node_infected_times[i]==0):
                continue                        

            a_hat=cp.Variable((self.num_nodes,))

            t_hat = cp.Variable((self.num_cascades,))

            all_zero_A = np.argwhere(np.all(A_potential[...,:]==0,axis=0))
            constraints = [a_hat[i[0]]==0 for i in all_zero_A]      
            c_act=0
            
            for c in range(self.num_cascades):
                if self.cascades[c][i]>0:                                  
                    cidx=self._find_node(self.sorted_cascades[c],i)                            
                    temp=[]
                    for j in range(0,cidx):                
                        temp.append(a_hat[self.sorted_cascades[c][j][0]])
                    constraints.append(t_hat[c_act]==cp.sum(temp))        

                    c_act+=1        

            constraints.append(a_hat>=0)                    
            obj=cp.Maximize(cp.sum(cp.multiply(-a_hat,(A_potential[:,i]+A_bad[:,i])))+cp.sum(cp.log(t_hat[:c_act])))

            prob = cp.Problem(obj,constraints)
            try:
                prob.solve(solver='SCS')
                result = [prob.status,prob.value,obj.value]
                results.append(result)
 
                total_obj+=obj.value
                A_hat[:,i]=a_hat.value
            except TypeError:
                A_hat[:i]=0
                errors.append(i)
      
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
            self.pow_survive(A)         
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



           