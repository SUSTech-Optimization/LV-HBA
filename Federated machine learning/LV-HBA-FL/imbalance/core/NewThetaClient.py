import torch
from torch.optim import SGD
import copy
from core.Client import Client
from core.function import *
# from utils.SGDC import SGDC
# from function import assign_correction_grad

class NewThetaClient(Client):
    def __init__(self, args, client_id, net, dataset=None, idxs=None, hyper_param=None, compared_param=None, if_inner=False) -> None:
        super().__init__(args, client_id, net, dataset, idxs, hyper_param, if_inner)
        self.init_net = copy.deepcopy(net)
        self.batch_num = len(self.ldr_train)
        self.compared_param = compared_param
        self.param_init = list(self.init_net.parameters())

    def loss_adjust_cross_entropy(self, logits, targets):
        loss = loss_adjust_cross_entropy(logits, targets, self.hyper_param)
        return loss

    def train_epoch(self, gama1, step, d2=False, ck=None):
        self.net.train()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=step)
    
        epoch_loss = []
        batch_loss = []
        # rand_idx = torch.randint(0,self.batch_num,())
        self.net.zero_grad()
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            # if batch_idx != rand_idx.item():
            #     continue
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            # print('now label is', labels,'length is',len(labels))
            log_probs = self.net(images)
            loss = self.loss_func(log_probs, labels)
            batch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            break
            
        self.net.zero_grad()
        i = 0
        lr = step if d2 else -step
        for k in self.net.parameters():
            d4_term3 = gama1*(self.param_init[i].detach() - self.compared_param[i].detach())
            k.data.add_(d4_term3, alpha=lr)
            i+=1

        if d2 and ck is not None:
            self.net.zero_grad()
            indirect_grad = self.grad_d_out_d_y_net(self.init_net)
            params = self.net.parameters()
            for i, p in enumerate(params):
                p.data.add_(indirect_grad[i], alpha=-ck)

        # term2 for d4

        
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return self.net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def grad_d_in_d_x(self, net=None, hp=None):
        if net is None:
            self.net0 = copy.deepcopy(self.net)
        else:
            self.net0 = copy.deepcopy(net)

        self.net0.train()
        hyper_param = self.hyper_param if hp is None else hp
        trainable_hp = get_trainable_hyper_params(hyper_param)
        num_weights = sum(p.numel() for p in trainable_hp)
        d_in_d_x = torch.zeros(num_weights, device=self.args.device)

        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(
                self.args.device), labels.to(self.args.device)
            self.net0.zero_grad()
            log_probs = self.net0(images)
            loss = self.loss_func(log_probs, labels)
            d_in_d_x += gather_flat_grad(grad(loss,
                                         trainable_hp, create_graph=True))
        d_in_d_x /= (batch_idx+1.)
        return d_in_d_x

    def hyper_grad_new(self, net_theta, hp_theta, ck):
        direct_grad = 0 #self.grad_d_out_d_x()
        indirect_grad = self.grad_d_in_d_x() - self.grad_d_in_d_x(net_theta, hp_theta)
        hyper_grad = ck*direct_grad + indirect_grad

        return hyper_grad

    def grad_d_out_d_y_net(self, net):
        self.net0 = copy.deepcopy(net)
        self.net0.train()
        params = [p for p in self.net0.parameters() if p.requires_grad==True]
        #num_weights = sum(p.numel() for p in params)
        d_out_d_y = []
        #d_out_d_y = torch.zeros(num_weights, device=self.args.device)
        for p in params:
            d_out_d_y.append(torch.zeros_like(p.detach()))
        for batch_idx, (images, labels) in enumerate(self.ldr_val):
            images, labels = images.to(
                self.args.device), labels.to(self.args.device)
            self.net0.zero_grad()
            log_probs = self.net0(images)
            loss = self.val_loss(log_probs, labels)
            d_out_d_y_batch = grad(loss, params, create_graph=True)
            for i, gd in enumerate(d_out_d_y):
                gd += d_out_d_y_batch[i]

        for gd in d_out_d_y:
            gd /= (batch_idx+1.)
        return d_out_d_y
