#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        if self.args.opt == 'ADAM':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=5e-4)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay = self.args.weight_decay)
        #optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
        #optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        #optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=5e-4)

        #print('train starts here!')
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                prev=net.state_dict()
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                current=net.state_dict()
                '''
                if self.args.verbose and batch_idx % 10 == 0: #self.args.verbose
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                '''
                batch_loss.append(loss.item())    
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return current, sum(epoch_loss) / len(epoch_loss)
    
class Scaffold(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        
    def train_net_scaffold(self, net, c_local, c_global):
    
        net.train()
        global_model = copy.deepcopy(net)
        # train and update
        if self.args.opt == 'ADAM':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=5e-4)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay = self.args.weight_decay)
        criterion = nn.CrossEntropyLoss().to(device)

        cnt = 0

    #writer = SummaryWriter()

        c_global_para = copy.deepcopy(c_global)  #.state_dict()
        c_local_para = copy.deepcopy(c_local)   #.state_dict()

        for epoch in range(epochs):
            epoch_loss_collector = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                #x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                images.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(images)
                loss = criterion(out, labels)

                loss.backward()
                optimizer.step()

                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        c_new_para = c_local.state_dict()
        c_delta_para = copy.deepcopy(c_local.state_dict())
        global_model_para = global_model.state_dict()
        net_para = net.state_dict()
        for key in net_para:
            c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
            c_delta_para[key] = c_new_para[key] - c_local_para[key]
        c_local.load_state_dict(c_new_para)
        return net_para, epoch_loss, c_new_para, c_delta_para

class ClusterDetect(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train2(self, net):
        net.train()
        # train and update
        if self.args.opt == 'ADAM':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=5e-4)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay = 5e-4)
        
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            net.zero_grad()
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss=(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), epoch_loss
