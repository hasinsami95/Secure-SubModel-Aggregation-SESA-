#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from torch import nn


def FedAvg_noscale(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1,len(w)):
            w_avg[k] += w[i][k]
        #w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedMult(w,sc):
    w_scaled = copy.deepcopy(w)
    for k in w_scaled.keys():
#         for i in range(1,len(w)):
#             w_scaled[k] = sc*w_scaled[k]
        w_scaled[k] = torch.mul(w_scaled[k], sc)
    return w_scaled

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1,len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
    
def FedAvg3(w,scale):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k]*scale[0]
        for i in range(1,len(w)):
            w_avg[k] += w[i][k]*scale[i]
        #w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvg2(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] =torch.div(w_avg[k],45)
    for k in w_avg.keys():
        for i in range(1,len(w)):
            if (i<5):
                w_avg[k] += torch.div(w[i][k],45)
            else:
                w_avg[k] += torch.div(w[i][k],(45/2))
        #w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAdd(w1,w2):
    w_final=copy.deepcopy(w1)
    #print(w2)
    for k in w_final.keys():
        w_final[k]=w1[k].to(torch.device("cuda:1"))+w2[k].to(torch.device("cuda:1"))
    return w_final


def FedSubstract(w1,w2):
    w_final=copy.deepcopy(w1)
    #print(w2)
    for k in w_final.keys():
        w_final[k]=w1[k].to(torch.device("cuda:1"))-w2[k].to(torch.device("cuda:1"))
    return w_final



def FedAvg_vectorization(w):
    vect=[]
    mat=[]
    dimension=[]
    w_avg = copy.deepcopy(w[0])
    count=0
    for k in w_avg.keys():
        for i in range(1,len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
        mat.append((w_avg[k].to(torch.device("cpu"))).numpy()) #converting weight tensor to matrix
        dimension.append(mat[count].shape) # storing the dimension of each matrix
        vect.append(mat[count].flatten())#vectorization of each matrix and appending them to vect list
        count=count+1
    
    return vect,dimension

def FedAvg_gradient(g):
    g_avg=copy.deepcopy(g[0])
    for i in range(1,len(g)):
        g_avg+=g[i]
    g_avg=g_avg/len(g)
    return g_avg
    
def FedAvg_gradient2(g):
    g_avg=copy.deepcopy(g[0])
    for i in range(1,len(g)):
        g_avg+=g[i]
    #g_avg=g_avg/len(g)
    return g_avg

def weight_vectorization(w):
    vect=[]
    mat=[]
    dimension=[]
    count=0
    for k in w.keys():
    
        mat.append((w[k].to(torch.device("cpu"))).numpy()) #converting weight tensor to matrix
        dimension.append(mat[count].shape) # storing the dimension of each matrix
        vect.append(mat[count].flatten())#vectorization of each matrix and appending them to vect list
        count=count+1
    feature_vect=np.concatenate([vect[0],vect[1],vect[2],vect[3]])
        #lets convert the feature vector into a column vector
    feature_vect=np.transpose(feature_vect.reshape(1,len(feature_vect)))
    return feature_vect,dimension
    #or return vect[0] depending on dimension (1 D or 2 D)
    
def weight_vectorization_gen(w):
    vect=[]
    mat=[]
    dimension=[]
    count=0
    feature_vect=np.array([])
    for k in w.keys():
    #for k in range(len(w)):
        mat.append((w[k].to(torch.device("cpu"))).numpy()) #converting weight tensor to matrix
        dimension.append(mat[count].shape) # storing the dimension of each matrix
        vect.append(mat[count].flatten())#vectorization of each matrix and appending them to vect list
        feature_vect=np.concatenate((feature_vect,vect[count]),axis=None)
        count=count+1
    #feature_vect=np.concatenate([vect[0],vect[1],vect[2],vect[3]])
        #lets convert the feature vector into a column vector
    feature_vect=np.transpose(feature_vect.reshape(1,len(feature_vect)))
    return feature_vect,dimension
    
def weight_vectorization_gen2(w):
    vect=[]
    mat=[]
    dimension=[]
    count=0
    feature_vect=np.array([])
    #for k in w.keys():
    for k in range(len(w)):
        mat.append((w[k].to(torch.device("cpu"))).numpy()) #converting weight tensor to matrix
        dimension.append(mat[count].shape) # storing the dimension of each matrix
        vect.append(mat[count].flatten())#vectorization of each matrix and appending them to vect list
        feature_vect=np.concatenate((feature_vect,vect[count]),axis=None)
        count=count+1
    #feature_vect=np.concatenate([vect[0],vect[1],vect[2],vect[3]])
        #lets convert the feature vector into a column vector
    feature_vect=np.transpose(feature_vect.reshape(1,len(feature_vect)))
    return feature_vect,dimension

def weight_vectorization_femnist(w):
    vect=[]
    mat=[]
    dimension=[]
    count=0
    for k in w.keys():
        mat.append((w[k].to(torch.device("cpu"))).numpy()) #converting weight tensor to matrix
        dimension.append(mat[count].shape) # storing the dimension of each matrix
        vect.append(mat[count].flatten())#vectorization of each matrix and appending them to vect list
        count=count+1
    feature_vect=np.concatenate([vect[0],vect[1],vect[2],vect[3],vect[4],vect[5]])
        #lets convert the feature vector into a column vector
    feature_vect=np.transpose(feature_vect.reshape(1,len(feature_vect)))
    return feature_vect,dimension
def weight_vectorization_cifar(w):
    #w=w.to(torch.device("cpu"))
    vect=[]
    mat=[]
    dimension=[]
    count=0
    for k in w.keys():
        mat.append((w[k].to(torch.device("cpu"))).numpy()) #converting weight tensor to matrix
        dimension.append(mat[count].shape) # storing the dimension of each matrix
        vect.append(mat[count].flatten())#vectorization of each matrix and appending them to vect list
        count=count+1
    feature_vect=np.concatenate([vect[0],vect[1],vect[2],vect[3],vect[4],vect[5],vect[6],vect[7],vect[8],vect[9],vect[10],vect[11],vect[12],
                                 vect[13],vect[14],vect[15]])
        #lets convert the feature vector into a column vector
    feature_vect=np.transpose(feature_vect.reshape(1,len(feature_vect)))
    return feature_vect,dimension
    
def vectorization(w):
    vect=[]
    mat=[]
    dimension=[]
    w_avg = copy.deepcopy(w)
    count=0
    for k in w_avg.keys():
        mat.append((w_avg[k]).numpy()) #converting weight tensor to matrix
        dimension.append(mat[count].shape) # storing the dimension of each matrix
        vect.append(mat[count].flatten())#vectorization of each matrix and appending them to vect list
        count=count+1
    return vect,dimension
        
def FedQAvg(w,q_val):
    w_avg = copy.deepcopy(w[0])
    #print(w_avg)
    for k in w_avg.keys():
        #print(k)
        for i in range(1,len(w)):
            #print(w[i][k])
            #print(torch.min(w[i][k]),torch.max(w[i][k]))
            #print((w[i][k]))
            temp = torch.mul(w[i][k], q_val).round()
            temp = torch.div(temp, q_val)
            
            #temp_np = temp.cpu().detach().numpy() 
            #print(k, np.shape(temp_np))
            #print((temp))
            w_avg[k] += temp
        w_avg[k] = torch.div(w_avg[k], len(w))
    
    loc = 0
    w_out = np.zeros((1,62346),dtype=float)
    return w_avg, w_out

def FedQAvg2(w,q_val,args):
    w_avg = copy.deepcopy(w[0])
    #print(w_avg)
    for k in w_avg.keys():
        #print(k)
        for i in range(1,len(w)):
            #print(w[i][k])
            #print(torch.min(w[i][k]),torch.max(w[i][k]))
            #print((w[i][k]))
            temp = w[i][k]
            temp_np = temp.cpu().detach().numpy()
            temp_np = np.round(temp_np*q_val,0)
            temp_np = temp_np / q_val
            
            #temp_np = temp.cpu().detach().numpy() 
            #print(k, np.shape(temp_np))
            #print((temp))
            w_avg[k] +=  torch.from_numpy(temp_np).float().to(args.device)
        w_avg[k] = torch.div(w_avg[k], len(w))
    
    loc = 0
    w_out = np.zeros((1,62346),dtype=float)
    '''
    for k in w_avg.keys():
        
        temp = w[i][k]
        temp_np = temp.cpu().detach().numpy()
            
        temp_len = np.prod(np.shape(temp_np))
        temp_np = np.reshape(temp_np,(temp_len))
        w_out[0][loc:loc+temp_len] = temp_np.astype(float)
            
        #print(i,k,w_out[i][loc],temp_np[0])
        loc = loc + temp_len
    '''
    return w_avg, w_out

def FedBrea(w_locals,m,A,q_bit,p,args):
    
    w_locals_np = {} # in finite-field
    for i in range(len(w_locals)):
        temp_np = {} #np.array([])
        w_local_np_flatten = np.array([],dtype=float)
    #     print(i)
        for k in w_locals[i].keys():
            tmp = w_locals[i][k].cpu().detach().numpy()
            tmp_flatten = tmp.flatten()
            w_local_np_flatten = np.concatenate((w_local_np_flatten,tmp_flatten), axis=0)
    #         tmp_q = my_q(tmp,8,p)
            temp_np.update({k: tmp})
    #         print(k,np.shape(tmp))
    #         temp_np = np.concatenate((temp_np,  ),axis=0)
    #     print(np.shape(w_local_np_flatten))
        w_locals_np.update({i: my_q(w_local_np_flatten, q_bit,p)})

    idx_select = multi_Krum(w_locals_np, m, A, q_bit, p)

    print('selected clients index;',idx_select)
    w_avg = copy.deepcopy(w_locals[idx_select[0]])
    #print('fc1.bias:',w_avg['fc1.bias'])

    for k in w_avg.keys():
        for i in range(1,m):
            q_val = (2**q_bit)
            temp = torch.mul(w_locals[idx_select[i]][k], q_val).round()
            temp = torch.div(temp, q_val)
            w_avg[k] += temp

            #temp = w_locals[i][k]
            #temp_np = temp.cpu().detach().numpy()
            #temp_np = np.round(temp_np*(2**q_bit),0)
            #temp_np = temp_np / (2**q_bit)
            
            #temp_np = temp.cpu().detach().numpy() 
            #print(k, np.shape(temp_np))
            #print((temp))
            #w_avg[k] +=  torch.from_numpy(temp_np).float().to(args.device)
        w_avg[k] = torch.div(w_avg[k], m)

    return w_avg, idx_select


def multi_Krum(w_locals_np, m, A, q_bit, p):
    
    # m = the number of selected models
    # A = the number of adversaries
    N = len(w_locals_np)
    dist = np.zeros((N,N))
#     dist_float = np.zeros((N,N)) # for debugging
    
    for i in range(N):
        for j in range(N):
            if i==j:
                dist[i,j] = 0
            else:
                dist_tmp = np.mod(w_locals_np[i] - w_locals_np[j],p)
                dist_tmp = np.reshape(dist_tmp,(len(dist_tmp),1))
                
                #dist_tmp_float = my_q_inv(dist_tmp, q_bit,p)
                dist[i,j] = np.mod(np.matmul(dist_tmp.transpose(),dist_tmp),p)
                #dist_float[i,j] = np.matmul(dist_tmp_float.transpose(),dist_tmp_float)
    #print(dist)
    #print(my_q_inv(dist, q_bit*2,p))
    #print(dist_float)
    
    # select lowest client whose socre is the lowest
    idx_select = []
    for k in range(m):
        score = np.zeros((N))
        
        for i in range(N):
            tmp_arr = dist[i,:]
            tmp_arr_sorted = np.sort(tmp_arr)
            num_sum = N-A-2-k
            score[i] = np.sum(tmp_arr_sorted[1:1+num_sum])
    #         print(tmp_arr)
    #         print(tmp_arr_sorted)
    #         print(score[i])
    #         print()

    #     print(score)
        idx_min_score = np.argmin(score)
        dist[idx_min_score,:] = (p-1)/2
        dist[:,idx_min_score] = (p-1)/2
        
        #print(dist)
        
        idx_select.append(idx_min_score)
        #print(idx_select)
    return idx_select


def Quantization(w):
    w_avg = copy.deepcopy(w[0])
    w_out = np.zeros((len(w),118346),dtype=float)
    for i in range(len(w)):
        loc = 0
        for k in w_avg.keys():
        
            temp = w[i][k]
            temp_np = temp.cpu().detach().numpy()
            
            temp_len = np.prod(np.shape(temp_np))
            temp_np = np.reshape(temp_np,(temp_len))
            w_out[i][loc:loc+temp_len] = temp_np.astype(float)
            
            #print(i,k,w_out[i][loc],temp_np[0])
            loc = loc + temp_len
            #print(loc)

    return np.round(w_out * 1024)

def Quantization_Finite(w,q_bit,p):
    w_avg = copy.deepcopy(w[0])
    w_out = np.zeros((len(w),62346),dtype=float)
    for i in range(len(w)):
        loc = 0
        for k in w_avg.keys():
        
            temp = w[i][k]
            temp_np = temp.cpu().detach().numpy()
            
            temp_len = np.prod(np.shape(temp_np))
            temp_np = np.reshape(temp_np,(temp_len))
            w_out[i][loc:loc+temp_len] = temp_np.astype(float)
            
            #print(i,k,w_out[i][loc],temp_np[0])
            loc = loc + temp_len
            print(loc)

    return my_q(w_out,q_bit,p) #w_out * 1024

def my_q(X,q_bit,p):
    X_int = np.round(X*(2**q_bit))
    is_negative = (abs(np.sign(X_int)) - np.sign(X_int))/2
    out = X_int + p * is_negative
    return out.astype('int64')

def my_q_inv(X_q,q_bit,p):
    flag = X_q - (p-1)/2
    is_negative = (abs(np.sign(flag)) + np.sign(flag))/2
    X_q = X_q - p * is_negative
    return X_q.astype(float)/(2**q_bit)

def my_score(w,m):
    # m = the number of output
    dist_array = np.zeros((len(w),len(w)))
    #print(np.shape(dist_array))

    for i in range(len(w)):
        vec_i = w[i][:]
        for j in range(len(w)):
            vec_j = w[j][:]
            temp_diff = vec_i - vec_j
            dist_array[i][j] = np.sum(np.multiply(temp_diff,temp_diff)) #temp_diff.transpose() * temp_diff
            #print(temp_dist)
    print(dist_array)

def my_score_Finite(w,m,q_bit,p):
    # m = the number of output
    dist_array = np.zeros((len(w),len(w)))
    #print(np.shape(dist_array))

    for i in range(len(w)):
        vec_i = w[i][:].astype('int64')
        for j in range(len(w)):
            vec_j = w[j][:].astype('int64')
            temp_diff = np.mod(vec_i - vec_j,p)
            #print(temp_diff)
            dist_array[i][j] = np.mod(np.sum(np.multiply(temp_diff,temp_diff)),p) #temp_diff.transpose() * temp_diff
            #print(temp_dist)
    #print(dist_array)
    out = my_q_inv(dist_array,0,p)
    print(out)

