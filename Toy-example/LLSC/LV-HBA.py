import numpy as np
import torch
import time
start_time = int(time.time())
# A_ = np.linalg.pinv(A)
def proj(x,A,b=None):
    I = torch.eye(x.shape[0])
    A_ = torch.linalg.pinv(A)
    if b is None:
        return torch.matmul(I-A_ @ A,x)
    else:
        return torch.matmul(I-A_ @ A,x)-A_@b
    
def F(x, y): 
    return torch.sin(c.T@x + d.T@y) + torch.log(torch.sum((x+y)**2) + 1)

def f(x, y):
    return 0.5 * torch.sum((x-y)**2)

def g(x, y):
    return A@y + H@x

def F_x(x, y):
    grad = torch.autograd.grad(F(x,y), x, allow_unused=True)[0]
    return grad if grad is not None else torch.zeros_like(x)
def F_y(x, y):
    grad = torch.autograd.grad(F(x,y), y, allow_unused=True)[0]
    return grad if grad is not None else torch.zeros_like(y)

def f_x(x, y):
    grad = torch.autograd.grad(f(x,y), x, allow_unused=True)[0]
    return grad if grad is not None else torch.zeros_like(x)

def f_y(x, y):
    grad = torch.autograd.grad(f(x,y), y, allow_unused=True)[0]
    return grad if grad is not None else torch.zeros_like(y)

def g_x(x, y):
    return H

def g_y(x, y):
    return A

def f_x_xhat_y(x, xhat, y):
    loss = f(x, y) - f(xhat.detach(), y)
    grad = torch.autograd.grad(loss, [x, y], allow_unused=True)
    return loss.detach().cpu().item(), grad[0], grad[1]

def fun(n, alpha=0.1, beta=0.1, eta=0.1, _lambda=1, gamma1=0.1, gamma2=1,  u=1000, seed=1):
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(10*np.ones(100))
    x.requires_grad_(True)
    x = proj(x, B)
    y = torch.from_numpy(10*np.ones(100))
    y = proj(y, A, H@x)
    y.requires_grad_(True)

    theta = torch.from_numpy(rng.random(100))
    theta = proj(theta, A, H@x)
    theta.requires_grad_(True)
    
    Z = torch.arange(0, u+1)
    z = torch.randint(u+1, (1,))
    z = torch.zeros(90).double()
    _lambda = torch.zeros(90).double()

    Ix = torch.eye(x.shape[0]).double()
    # fix y_opt and delete this
    C=Ix

    res = []
    time_computation=[]
    print((g(x, theta)).shape)
    algorithm_start_time = time.time()
    for k in range(n):
        # clac d4
        ck= (k+1)**0.3

        d4_0 = f_y(x, theta) + _lambda@A  + (theta - y) / gamma1
        d4_1 = - g(x, theta) + (_lambda - z) / gamma2
        # update theta, lambda
        theta -= (eta * d4_0)
        _lambda -= (eta * d4_1)
        # proj
            
        theta = proj(theta, A, H@x)
        
        # calc d1 d2 d3, and update x, y, z respectively
        xk = x.clone().detach().requires_grad_(True)
        xk2 = x.clone().detach().requires_grad_(True)
        xk3 = x.clone().detach().requires_grad_(True)
        d1 = F_x(x, y) / ck + f_x(xk, y) - f_x(xk2, theta) - _lambda@g_x(xk3, theta)
        d2 = F_y(x, y) / ck + f_y(xk, y) - (y - theta) / gamma1
        with torch.no_grad():
            x -= alpha * d1
            y -= alpha * d2
            x1 = torch.cat([x,y],0)
            Ox_H = torch.zeros_like(H)
            Ox_A = torch.zeros_like(A)
            U = torch.cat([H,A], 1)
            V = torch.cat([B,Ox_A],1)
            P = torch.cat([U,V],0)
            x1 = proj(x1,P)
            x, y = torch.split(x1, len(x1) //2)
        x = torch.tensor(x,requires_grad=True)
        y = torch.tensor(y,requires_grad=True)
        d3 = - (_lambda - z) / gamma2
        t_z = z - (beta * d3)
        z = t_z
        y_opt = C@((Ix-torch.linalg.pinv(A@C)@(A@C))@x-torch.linalg.pinv(A@C)@(H@x))
        df_dy = torch.cos(c@x+d@y)*(d.T)+1/(x.dot(x)+y.dot(y)+1) * y
        df_dx = torch.cos(c@x+d@y)*c+2/(torch.square(torch.norm(x+y))+1) * (x+y)
        #res.append((torch.linalg.norm(y-y_opt,2)).detach().numpy()) # y_gap
        time_computation.append(time.time()-algorithm_start_time)
        #res.append((torch.square(torch.linalg.norm(df_dx,2))).detach().numpy()) # norm
        res.append((F(x,y)).detach().numpy())
        #print(d3)
        #print(z)
    return res, time_computation

if __name__ == '__main__':
    from pathlib import Path
    import os
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory

    # while True:
    #     A = np.random.rand(90,100)
    #     rank = np.linalg.matrix_rank(A)
    #     if rank == A.shape[0] and rank < A.shape[1]:
    #         print('A rank is', rank)
    #         np.save(os.path.join(str(ROOT), 'A.npy'), A)
    #         break

    # while True:
    #     B = np.random.rand(45,100)
    #     Z = np.zeros(B.shape)
    #     B = np.concatenate([B,Z],axis=0)
    #     rank = np.linalg.matrix_rank(B)
    #     if rank < B.shape[0]:
    #         print('B rank is', rank)
    #         np.save(os.path.join(str(ROOT), 'B.npy'), B)
    #         break

    # h = np.random.rand(90,100)
    # np.save(os.path.join(str(ROOT), 'h.npy'), h)
    A = torch.from_numpy(np.load(os.path.join(str(ROOT), 'A.npy')))
    B = torch.from_numpy(np.load(os.path.join(str(ROOT), 'B.npy')))
    H = torch.from_numpy(np.load(os.path.join(str(ROOT), 'h.npy')))
    # C = np.load('C.npy')
    c = torch.from_numpy(np.load(os.path.join(str(ROOT), 'D.npy')))
    d = torch.from_numpy(np.load(os.path.join(str(ROOT), 'E.npy')))
    res, tc = fun(1000, alpha=0.02, beta=0.001, eta=0.1, _lambda=0.1, gamma1=1, gamma2=1, u=1000, seed=1)
    tmp_time = int(time.time())
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    ax = plt.gca()
    plt.plot(tc,res,'-',label="Gaps")
    plt.xlabel('Running time /s')
    plt.ylabel("y gap")
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.show()