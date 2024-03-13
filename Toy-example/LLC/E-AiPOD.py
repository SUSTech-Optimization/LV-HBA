import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

import numpy as np
import scipy as sp
import yaml
import time
import torch
import scipy.linalg
start_time = int(time.time())

# while True:
#     A = np.random.rand(90,100)
#     rank = np.linalg.matrix_rank(A)
#     if rank == A.shape[0] and rank < A.shape[1]:
#         print('A rank is', rank)
#         np.save('A', A)
#         break

# while True:
#     B = np.random.rand(80,100)
#     Z = np.zeros(B.shape)
#     B = np.concatenate([B,Z],axis=0)
#     rank = np.linalg.matrix_rank(B)
#     if rank < B.shape[0]:
#         print('B rank is', rank)
#         np.save('B', B)
#         break

# while True:
#     C = np.random.rand(100,100)
#     rank = np.linalg.matrix_rank(C)
#     if rank == C.shape[0] and rank == C.shape[1]:
#         print('C rank is', rank)
#         np.save('C', C)
#         break
    
# D = np.random.rand(100)
# np.save('D', D)
# E = np.random.rand(100)
# np.save('E', E)
# h = np.random.rand(90,100)
# np.save('h', h)

class Logger():
    def __init__(self, filename) -> None:
        self.filename=filename
        self.logs={"k": [], "norm": [], "ygap":[], "proj count":[]}
        self.counter=0
    def logging(self,k,count,ygap,norm):
        self.logs["k"].append(k)
        self.logs["proj count"].append(count)
        self.logs["ygap"].append(ygap)
        self.logs['norm'].append(norm)
        self.counter+=1
    def save(self):
        #if self.counter%100==0:
        f = open(self.filename, mode="w+")
        yaml.dump(self.logs, f)
        f.close()
        
def proj(x, A, b=None):
    if A.ndim < 2:
        raise ValueError("Matrix A must be at least 2-dimensional.")
    I = np.eye(x.shape[0])
    A1 = A @ A.T
    A_ = np.linalg.pinv(A1)
    if b is None:
        return np.matmul(I - A.T @ A_ @ A, x)
    else:
        return np.matmul(I - A.T @ A_ @ A, x) - A.T @ A_ @ b

def proj1(x, b = None):
    if b is None:
        return x - sum(x) / len(x)
    else:
        return x - (sum(x) + b) / len(x)

if __name__ == '__main__':
    x = 10*np.ones((2000,1))
    y = 10*np.ones((2000,1))
    e1 = np.ones((2000,1))
    e2 = np.ones((4000,1))
    E = np.eye(2000)
    O = np.zeros((2000,2000))
    A = np.concatenate((E, O), axis=1)
    B = np.concatenate((O, E), axis=1)
    A_ = np.linalg.pinv(e2.T)


    K=150
    S=5
    T=2
    lr=0.00005
    hlr=0.002
    p=1
    sig=0.1
    seed=1
    
    logs = Logger(f'./alsetskip_K(p=1){K}_S{S}_T{T}_lr{lr}_hlr{hlr}_p{p}_sig{sig}_seed{seed}_{start_time}.yaml') 
    rng = np.random.default_rng(seed)
    x = 10*np.ones((2000,1))
    y1 = 10*np.ones((2000,1))
    y2 = 10*np.ones((2000,1))
    y = np.concatenate((y1, y2), axis=0)
    x = proj1(x,e2.T@y)
    y = proj1(y,e1.T@x)
    y1 = A@y
    y2 = B@y
    r = np.zeros(y.shape)
    Ix = np.eye(x.shape[0])
    # fix y_opt and delete this
    proj_count = 0

       
    # compute metric at first round
    y_opt = 0.7*e2
    y_gap = np.linalg.norm(y-y_opt,2)
    v=sp.linalg.null_space(e2.T)
    df_dy = -(A.T@(x-A@y))+B.T@(B@y-e1)
    dg_dyy= A.T@A
    temp = np.linalg.pinv(v.T@dg_dyy@v)
    w = v@temp@(v.T)@df_dy
    df_dx=x-A@y
    dh = e1
    dg_dxy= A
    dg_dyy= A.T@A
    hg = (dh @ (A_.T) @ dg_dyy - dg_dxy) @ w
    dF = df_dx+hg
    norm =np.square(np.linalg.norm(dF,2))
    F = 0.5*np.linalg.norm(x-A@y,2)**2+0.5*np.linalg.norm(B@y-e1,2)**2
    
    # print('step',0,'time','proj',0,'norm {:.2f}'.format(norm),'y gap {:.2f}'.format(y_gap))

    logs.logging(0,0,y_gap.item(),norm.item())
    logs.save()
    
    Kp = np.min((int(K/p),4000))

    list_k_time=np.array([0])

    for k in range(Kp):
        for s in range(S):
            dg_dy= A.T@A@y-A.T@x+A.T@e1
            # n = rng.normal(0,sig,dg_dy.shape) 
            z=y-lr*(dg_dy-r)
            theta = rng.binomial(1,p)
            if theta==1:
                y = z-lr/p * r
                y = proj1(y,e1.T@x)
                r=r+p/lr * (y-z)
            else:
                y=z
            proj_count+=theta
        # this is problematic when C is not I
        y1_opt = 0.7*e1
        y2_opt = -0.4*e1
        df_dy = -(A.T@(x-A@y))+B.T@(B@y-e1)
        # df_dy+=rng.normal(0,sig,df_dy.shape)
        dg_dyy= A.T@A
        # dg_dyy+=rng.normal(0,sig,dg_dyy.shape)
        temp = np.linalg.pinv(v.T@dg_dyy@v)
        w = v@temp@(v.T)@df_dy
        for t in range(T):
            df_dx=x-A@y
            # df_dx+=rng.normal(0,sig,df_dx.shape)
            dh = e1
            # dh+=rng.normal(0,sig,dh.shape)
            dg_dxy= A
            # dg_dyy+=rng.normal(0,sig,dg_dxy.shape)
            dg_dyy=A.T@A
            # dg_dyy+=rng.normal(0,sig,dg_dyy.shape)
            hg = (dh @ (A_.T) @ dg_dyy - dg_dxy) @ w
            dF = df_dx+hg
            # n=rng.normal(0,sig,dg_dy.shape)
            x = x - hlr*(dF)
        y_opt = np.concatenate((y1_opt, y2_opt), axis=0)
        x_opt = -0.3*e1
        #y_gap = (np.linalg.norm(y-y_opt,2))
        y_gap = np.log((np.linalg.norm(x-x_opt,2)) / (np.linalg.norm(x_opt,2)))
        v=sp.linalg.null_space(e2.T)
        norm =np.linalg.norm(dF,2)
        list_k_time=np.append(list_k_time,time.time()-start_time)
        # print('step',k+1,'proj',proj_count,'norm {:.2f}'.format(norm),'y gap {:.2f}'.format(y_gap))
        logs.logging(k+1,proj_count,y_gap.item(),norm.item())
        logs.save()
        print(y_gap)
    tmp_time = int(time.time())
    print(f'end time {tmp_time-start_time}')
    import matplotlib.pyplot as plt
    with open(logs.filename, 'r') as f:
        logs_dict = yaml.load(f, Loader=yaml.FullLoader)
        list_k = logs_dict.get('k')
        list_ygap = logs_dict.get('ygap')
        #list_norm = logs_dict.get('norm')
        # plt.plot(list_k,np.log(list_ygap))
        plt.plot(list_k_time,list_ygap)
        plt.xlabel('time(s)')
        plt.ylabel(r'$\log{ygap}$')
        ## plt.plot(list_k,np.log(list_norm))
        plt.show()
        
        