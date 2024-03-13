#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import yaml
import time
from core.test import test_img
from utils.Fed import FedAvg, FedAvgGradient
from models.SvrgUpdate import LocalUpdate
from utils.options import args_parser
from utils.dataset import load_data
from models.ModelBuilder import build_model
from core.NewClientManage import NewClientManage
from utils.my_logging import Logger
from core.function import assign_hyper_gradient
from torch.optim import SGD
import torch

import numpy as np
import copy

start_time = int(time.time())

if __name__ == '__main__':

    args = args_parser()


    dataset_train, dataset_test, dict_users, args.img_size, dataset_train_real = load_data(args)

    net_glob = build_model(args)
    net_glob_theta = copy.deepcopy(net_glob)


    w_glob = net_glob.state_dict()
    w_glob_theta = net_glob_theta.state_dict()


    if args.output == None:
        logs = Logger(f'./save/imbafed_IO{args.optim}_{args.dataset}_sz{args.size}_iid{args.iid}_\
K{args.num_users}_C{args.frac}_{args.model}_E{args.epochs}_\
lr{args.lr}_hlr{args.hlr}_B{args.local_bs}_tau{args.outer_tau}_blo{not args.no_blo}_\
IE{args.inner_ep}_p{args.p}_N{args.neumann}_{args.hvp_method}_s{args.seed}_{start_time}.yaml')  
    else:
        logs = Logger(args.output)                                                           
    

    mu=0.1**(1/9)
    probability=np.array([mu**-i for i in range(0,10)])
    wy=probability/np.linalg.norm(probability)
    ly= np.log(1./probability)
    hyper_param={
            'dy':torch.zeros(args.num_classes, requires_grad=True, device = args.device),
            'ly':torch.zeros(args.num_classes, requires_grad=True, device = args.device),
            'wy':torch.tensor(wy, device = args.device, dtype = torch.float32)
            }

    comm_round=0

    const_vars = {
        'alpha': 0.1,
        'beta': 0.1,
        'yita': 0.1, 
        'gama': [0.1, 0.1],
        'ck': 0.1,
    }
    alpha = const_vars['alpha']
    hyper_optimizer=SGD([hyper_param[k] for k in hyper_param], lr=alpha)
    

    nets_loc = [copy.deepcopy(net_glob) for i in range(args.num_users)]
    nets_loc_theta = [copy.deepcopy(net_glob) for i in range(args.num_users)]


    m = max(int(args.frac * args.num_users), 1)
    client_idx = np.random.choice(range(args.num_users), m, replace=False)
    client_manage=NewClientManage(args, net_glob, net_glob_theta, client_idx, const_vars, dataset_train, dict_users, hyper_param, nets_loc, nets_loc_theta)
    for iter in range(args.epochs):
        start_time = time.time()
        ck = 1/((iter+1)**0.5) 
        client_manage.update_theta_with_d4() 
        client_manage.update_x_with_d1(ck) 
        client_manage.update_y_with_d2(ck) 


        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train_real, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        end_time = time.time()
        print("Test acc/loss: {:.2f} {:.2f}".format(acc_test, loss_test),
              "Train acc/loss: {:.2f} {:.2f}".format(acc_train, loss_train),
              f"Comm round: {comm_round}",
              "time: {:.2f}s".format(end_time - start_time))

        logs.logging(np.array([-1]), acc_test, acc_train, loss_test, loss_train, comm_round)
        logs.save()
        comm_round += 1

        if args.round>0 and comm_round>args.round:
            break
        
