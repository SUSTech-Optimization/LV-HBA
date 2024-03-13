import copy
from cv2 import log
import numpy as np

import torch
from torch.optim import SGD

from utils.Fed import *
from core.function import *
from core.SGDClient_hr import SGDClient
from core.SVRGClient_hr import SVRGClient
from core.NewThetaClient import NewThetaClient
from core.Client_hr import Client
from core.ClientManage import ClientManage

class NewClientManage():
    def __init__(self,args, net_glob, net_glob_theta, client_idx, const_vars, dataset, dict_users, hyper_param, nets_loc, nets_loc_theta) -> None:
        self.net_glob=net_glob
        self.client_idx=client_idx
        self.args=args
        self.dataset=dataset
        self.dict_users=dict_users
        self.p = torch.tensor(self.args.p)

        self.nets_loc = nets_loc
        self.net_glob_theta = net_glob_theta
        
        self.hyper_param = copy.deepcopy(hyper_param)
        alpha = const_vars['alpha']
        self.hyper_optimizer= SGD([self.hyper_param[k] for k in self.hyper_param], lr=alpha)
        
        self.nets_loc = nets_loc
        self.nets_loc_theta = nets_loc_theta

        self.const_vars = const_vars

    def update_theta_with_d4(self):
        w_glob = self.net_glob_theta.state_dict()
        nets_loc = self.nets_loc_theta
        y_param = list(self.net_glob.parameters())
        gama = self.const_vars['gama']
        gama1 = gama[0]
        yita = self.const_vars['yita']
        w_locals=[]

        for idx in self.client_idx:
            client = NewThetaClient(self.args, idx, nets_loc[idx], self.dataset, self.dict_users, self.hyper_param, y_param)
            w, loss = client.train_epoch(gama1, yita)
            w_locals.append(copy.deepcopy(w))

        w_glob = FedAvg(w_locals)
        for idx in self.client_idx:
            net_loc = self.nets_loc_theta[idx]
            net_loc.load_state_dict(w_glob)
        self.net_glob_theta.load_state_dict(w_glob)

    def update_x_with_d1(self, ck):
        client_locals=[]
        w_glob = self.net_glob.state_dict()
        nets_loc = self.nets_loc
        for idx in self.client_idx:
            client = NewThetaClient(self.args, idx, nets_loc[idx], self.dataset, self.dict_users, self.hyper_param)
            client_locals.append(client)

        hg_locals =[]
        for client in client_locals:
            hg = client.hyper_grad_new(
                self.nets_loc_theta[client.client_id], 
                client.hyper_param, ck)
            hg_locals.append(hg)
        hg_glob=FedAvgP(hg_locals, self.args)

        hg_locals =[]
        for client in client_locals:
            for _ in range(self.args.outer_tau):
                h = client.hyper_svrg_update(hg_glob)
            hg_locals.append(h)
        hg_glob=FedAvgP(hg_locals, self.args)

        assign_hyper_gradient(self.hyper_param, hg_glob)
        self.hyper_optimizer.step()

    def update_y_with_d2(self, ck):
        w_glob = self.net_glob.state_dict()
        nets_loc = self.nets_loc
        theta_param = list(self.net_glob_theta.parameters())
        gama = self.const_vars['gama']
        gama1 = gama[0]
        alpha = self.const_vars['alpha']
        w_locals=[]
        for idx in self.client_idx:
            client = NewThetaClient(self.args, idx, nets_loc[idx], self.dataset, self.dict_users, self.hyper_param, theta_param)
            w, loss = client.train_epoch(gama1, alpha, d2=True, ck=ck)
            w_locals.append(copy.deepcopy(w))
        w_glob = FedAvg(w_locals)

        for idx in self.client_idx:
            net_loc = self.nets_loc[idx]
            net_loc.load_state_dict(w_glob)
        self.net_glob.load_state_dict(w_glob)
            


    
