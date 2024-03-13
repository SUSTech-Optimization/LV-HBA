import cvxpy as cp
import numpy as np
import time
import torch
import copy
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.nn import functional as F

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, random, time
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class LinearSVM(nn.Module):
    def __init__(self, input_size, n_classes, n_sample):
        super(LinearSVM, self).__init__()
        self.w = nn.Parameter(torch.ones(n_classes, input_size))
        self.b = nn.Parameter(torch.tensor(0.))
        self.xi = nn.Parameter(torch.ones(n_sample))
        self.C = nn.Parameter(torch.ones(n_sample))
    
    def forward(self, x):
        return F.linear(x, self.w, self.b)

    def loss_upper(self, y_pred, y_val):
        y_val_tensor = torch.Tensor(y_val)
        x = torch.reshape(y_val_tensor, (y_val_tensor.shape[0],1)) * y_pred / torch.linalg.norm(self.w)
        relu = nn.LeakyReLU()
        loss = torch.sum(relu(2*torch.sigmoid(-5.0*x)-1.0))
        return loss

    def loss_lower(self):
        w2 = 0.5*torch.linalg.norm(self.w)**2
        c_exp=torch.exp(self.C)
        xi_term = 0.5 * (torch.dot(c_exp, (self.xi)**2))
        loss =  w2 + xi_term
        return loss

    def constrain_values(self, srt_id, y_pred, y_train):
        xi_sidx = srt_id
        xi_eidx = srt_id+len(y_pred)
        xi_batch = self.xi[xi_sidx:xi_eidx]
        return 1-xi_batch-y_train.view(-1)*y_pred.view(-1)

def run(seed, epochs):
    print("========run seed: {}=========".format(seed))
    data_list=[]

    f = open("diabete.txt",encoding = "utf-8")
    a_list=f.readlines()
    f.close()
    for line in a_list:
        line1=line.replace('\n', '')
        line2=list(line1.split(' '))
        y=float(line2[0])
        x= [float(line2[i].split(':')[1]) for i in (1,2,3,4,5,6,7,8)]
        data_list.append(x+[y])

    data_array=np.array(data_list)
    np.random.seed(seed)
    np.random.shuffle(data_array)

    z_train=data_array[:500, :-1]
    y_train=data_array[:500, -1]
    z_val=data_array[500:650, :-1]
    y_val=data_array[500:650, -1]
    z_test=data_array[650:, :-1]
    y_test=data_array[650:, -1]

    batch_size = 256
    data_train = TensorDataset(
        torch.tensor(z_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(
        dataset=data_train,
        batch_size=batch_size,
        shuffle=True)
    data_val = TensorDataset(
        torch.tensor(z_val, dtype=torch.float32), 
        torch.tensor(y_val, dtype=torch.float32))
    val_loader = DataLoader(
        dataset=data_val,
        batch_size=batch_size,
        shuffle=True)
    data_test = TensorDataset(
        torch.tensor(z_test, dtype=torch.float32), 
        torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(
        dataset=data_test,
        batch_size=batch_size,
        shuffle=True)


    #print(y_train.shape)
    #print(y_val.shape)
    #print(y_test.shape)
    #print(c_array_tensor.shape)

   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature = 8


    N_sample = y_train.shape[0]
    model = LinearSVM(feature, 1, N_sample)
    model.C.data.copy_(torch.Tensor(z_train.shape[0]).uniform_(-6.0,-5.0))
    model_theta = copy.deepcopy(model)

    alpha = 0.01
    beta = 0.1
    yita = 0.00 
    gama1 = 0.1
    gama2 = 0.1
    #ck = 0.1
    u = 200

    N_sample = y_train.shape[0]
    lamda = torch.ones(N_sample) #+ 1./N_sample
    z = torch.ones(N_sample) #+ 1./N_sample

    params = [p for n, p in model.named_parameters() if n != 'C']
    params_theta = [p for n, p in model_theta.named_parameters() if n != 'C']

    
    x = cp.Variable(feature+1+2*N_sample)
    y = cp.Parameter(feature+1+2*N_sample)
    w = x[0:feature]
    b = x[feature]
    xi = x[feature+1:feature+1+N_sample]
    C = x[feature+1+N_sample:]

    loss = cp.norm(x-y, 2)**2

    constraints=[]
    for i in range(y_train.shape[0]):
        constraints.append(1 - xi[i] - y_train[i] * (cp.scalar_product(w, z_train[i])+b) <= 0)


    obj = cp.Minimize(loss)
    prob = cp.Problem(obj, constraints)


    val_loss_list=[]
    test_loss_list=[]
    val_acc_list=[]
    test_acc_list=[]
    time_computation=[]
    
    #epochs = 80
    algorithm_start_time=time.time()
    for k in range(epochs):
        ck = 1/((k+1)**0.3)
        
        model_theta.zero_grad()
        loss = model_theta.loss_lower()
        loss.backward()

        idx_glob = 0
        constr_glob_list = torch.ones(0)
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            log_probs = model_theta(images)
            cv = model_theta.constrain_values(idx_glob, log_probs, labels)
            lamda_batch = lamda[idx_glob:idx_glob+len(labels)]
            cv.backward(lamda_batch)
            constr_glob_list = torch.cat((constr_glob_list, cv), 0)
            idx_glob += len(labels)


        for i, p_theta in enumerate(params_theta):
            d4_theta = torch.zeros_like(p_theta.data)
            if p_theta.grad is not None:
                d4_theta += p_theta.grad
            d4_theta += gama1*(p_theta.data - params[i].data)
            p_theta.data.add_(d4_theta, alpha=-yita)

        lamda = lamda - yita*(-constr_glob_list + gama2*(lamda - z))
        lamda = torch.clamp(lamda,0,u)


        model_theta.zero_grad()

        loss = model_theta.loss_lower()
        loss.backward()

        idx_glob = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            log_probs = model_theta(images)
            cv = model_theta.constrain_values(idx_glob, log_probs, labels)
            lamda_batch = lamda[idx_glob:idx_glob+len(labels)]
            cv.backward(lamda_batch)
            idx_glob += len(labels)


        model.zero_grad()

        loss = model.loss_lower()
        loss.backward()

        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            log_probs = model(images)
            loss = model.loss_upper(log_probs, labels)
            loss.backward(torch.tensor(ck))
        

        for i, p in enumerate(params):
            d2 = torch.zeros_like(p.data)
            if p.grad is not None:
                d2 += p.grad
            d2 += gama1*(params_theta[i].data - p.data)
            p.data.add_(d2, alpha=-alpha)

        d1 = model.C.grad - model_theta.C.grad
        model.C.data.add(d1, alpha=-alpha)

        
        #prob.solve(solver='MOSEK', warm_start=True, verbose=True)
        y_w = model.w.data.view(-1).detach().numpy()
        y_b = model.b.data.detach()
        y_xi =  model.xi.data.view(-1).detach().numpy()
        y_C = model.C.data.view(-1).detach().numpy()

        y.value = np.concatenate((y_w, np.array([y_b]), y_xi, y_C))

        prob.solve(solver='ECOS', abstol=2e-3,reltol=2e-3,max_iters=1000000000, warm_start=True)  
        C_solv = torch.Tensor(np.array(C.value))
        w_solv = torch.Tensor(np.array([w.value]))
        b_solv = torch.tensor(b.value)
        xi_solv = torch.Tensor(np.array(xi.value))

        model_theta.C.data.copy_(C_solv)
        model.C.data.copy_(C_solv)
        model.w.data.copy_(w_solv)
        model.b.data.copy_(b_solv)
        model.xi.data.copy_(xi_solv)
        d3 = -gama2*(lamda-z)
        z += -beta*d1
        z = torch.clamp(z,0,u)

        number_right = 0
        val_loss = 0
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            log_probs = model(images)
            for i in range(len(labels)):
                q=log_probs[i]*labels[i]
                if q>0:
                    number_right=number_right+1
            val_loss += model.loss_upper(log_probs, labels)
        val_acc=number_right/len(y_val)
        val_loss /= 15.0

        number_right=0
        test_loss = 0
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            log_probs = model(images)
            for i in range(len(labels)):
                q=log_probs[i]*labels[i]
                if q>0:
                    number_right=number_right+1
            test_loss += model.loss_upper(log_probs, labels)
        test_acc=number_right/len(y_test)
        test_loss /= 11.80
        print("val acc: {:.2f}".format(val_acc),
              "val loss: {:.2f}".format(val_loss),
              "test acc: {:.2f}".format(test_acc),
              "test loss: {:.2f}".format(test_loss),
              "round: {}".format(k))

        val_loss_list.append(val_loss.detach().numpy())
        test_loss_list.append(test_loss.detach().numpy())
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        time_computation.append(time.time()-algorithm_start_time)

    end_time = time.time()
    time_duaration = end_time - algorithm_start_time

    return val_loss_list,test_loss_list,val_acc_list,test_acc_list,time_computation,time_duaration



if __name__ == "__main__":
    if len(sys.argv) == 3:
        data_loop = int(sys.argv[1])
        epochs = int(sys.argv[2])
    else:
        print("Invalid params, run with default setting")
        data_loop = 40
        epochs = 80
    start_time = time.time()
    val_loss_array=[]
    test_loss_array=[]
    val_acc_array=[]
    test_acc_array=[]
    time_duaration_array=[]
    for seed in range(1,data_loop):
        val_loss_list,test_loss_list,val_acc_list,test_acc_list,time_computation, time_duaration=run(seed, epochs)
        val_loss_array.append(np.array(val_loss_list))
        test_loss_array.append(np.array(test_loss_list))
        val_acc_array.append(np.array(val_acc_list))
        test_acc_array.append(np.array(test_acc_list))
        time_computation=np.array(time_computation)
        time_duaration_array.append(time_duaration)
    val_loss_array=np.array(val_loss_array)
    test_loss_array=np.array(test_loss_array)
    val_acc_array=np.array(val_acc_array)
    test_acc_array=np.array(test_acc_array)
    time_duaration_array=np.array(time_duaration_array)

    val_loss_mean=np.sum(val_loss_array,axis=0)/val_loss_array.shape[0]
    val_loss_sd=np.sqrt(np.var(val_loss_array,axis=0))/2.0
    test_loss_mean=np.sum(test_loss_array,axis=0)/test_loss_array.shape[0]
    test_loss_sd=np.sqrt(np.var(test_loss_array,axis=0))/2.0

    val_acc_mean=np.sum(val_acc_array,axis=0)/val_acc_array.shape[0]
    val_acc_sd=np.sqrt(np.var(val_acc_array,axis=0))/2.0
    test_acc_mean=np.sum(test_acc_array,axis=0)/test_acc_array.shape[0]
    test_acc_sd=np.sqrt(np.var(test_acc_array,axis=0))/2.0

    time_mean=np.sum(time_duaration_array,axis=0)/time_duaration_array.shape[0]
    print("*******************")
    print("Average runing time for my algorithm: ", time_mean)
    print("Average test loss: ", test_loss_mean[-1])
    print("Average test acc: ", test_acc_mean[-1])
    print("*******************")

    plt.rcParams.update({'font.size': 18})
    #plt.rcParams['font.sans-serif']=['Arial']
    plt.rcParams['axes.unicode_minus']=False 
    #axis=np.arange(1, 81)
    axis=time_computation
    plt.figure(figsize=(8,6))
    #plt.grid(linestyle = "--") 
    ax = plt.gca()
    plt.plot(axis,val_loss_mean,'-',label="Validation loss")
    ax.fill_between(axis,val_loss_mean-val_loss_sd,val_loss_mean+val_loss_sd,alpha=0.2)
    plt.plot(axis,test_loss_mean,'--',label="Test loss")
    ax.fill_between(axis,test_loss_mean-test_loss_sd,test_loss_mean+test_loss_sd,alpha=0.2)
    #plt.xticks(np.arange(0,iterations,40))
    plt.title('Liner SVM')
    plt.xlabel('Running time /s')
    #plt.legend(loc=4)
    plt.ylabel("Loss")
    #plt.xlim(-0.5,3.5)
    #plt.ylim(0.5,1.0)
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    #plt.setp(ltext, fontsize=18,fontweight='bold')
    plt.savefig('new_run_diabete_1.pdf') 
    plt.show()

    #axis=np.arange(1, 81)
    axis=time_computation
    plt.figure(figsize=(8,6))
    ax = plt.gca()
    plt.plot(axis,val_acc_mean,'-',label="Validation accuracy")
    ax.fill_between(axis,val_acc_mean-val_acc_sd,val_acc_mean+val_acc_sd,alpha=0.2)
    plt.plot(axis,test_acc_mean,'--',label="Test accuracy")
    ax.fill_between(axis,test_acc_mean-test_acc_sd,test_acc_mean+test_acc_sd,alpha=0.2) 
    #plt.xticks(np.arange(0,iterations,40))
    plt.title('Liner SVM')
    plt.xlabel('Running time /s')
    plt.ylabel("Accuracy")
    plt.ylim(0.64,0.8)
    #plt.legend(loc=4)
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    #plt.setp(ltext, fontsize=18,fontweight='bold') 
    plt.savefig('new_run_diabete_2.pdf') 
    plt.show()

    end_time = time.time()
    print("time", end_time - start_time)

