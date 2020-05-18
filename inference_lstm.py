# -*- coding: utf-8 -*-
"""
Created on Sun May 17 22:04:30 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 17 18:13:19 2020

@author: Administrator
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from lstmdataset import InfectDataset,data_split,inferece_data_split
from dataloader import DataLoader
from lstm import RNN
import argparse
if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_num', type=int, default=392)
    parser.add_argument('--feat_dim', type=int, default=1)
    parser.add_argument('--n_his', type=int, default=5)
    parser.add_argument('--n_pred', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save', type=int, default=10)
    parser.add_argument('--Ks', type=int, default=3)  #equal to num_layers
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--keep_prob', type=float, default=1.0)
    parser.add_argument('--opt', type=str, default='ADAM')
    parser.add_argument('--inf_mode', type=str, default='sep')
    parser.add_argument('--input_file', type=str, default='/home/ubuntu/baidu/data_processed/region_migration.csv')
    parser.add_argument('--label_file', type=str, default='/home/ubuntu/baidu/data_processed/infection.csv')
    parser.add_argument('--adj_mat_file', type=str, default='/home/ubuntu/baidu/data_processed/adj_matrix.npy')
    parser.add_argument('--output_path', type=str, default='./outputs/')
    parser.add_argument('--val_num', type=str, default=3)
    parser.add_argument('--test_num', type=str, default=1)
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--train_all', action='store_true')
    
    parser.add_argument('--region_names_file', type=str, 
            default='/home/ubuntu/baidu/data_processed/region_names.txt')
    args = parser.parse_args()
    dataset = InfectDataset(args)

    inferece_test=inferece_data_split(dataset, args)
#    train_loader = DataLoader(train, batch_size=1, shuffle=False)
#    eval_loader=DataLoader(valid, batch_size=1, shuffle=False)
    inferece_loader=DataLoader(inferece_test, batch_size=1, shuffle=False)
    
    rnn=RNN(3,128,args.n_his,1).cuda()
    rnn.load_state_dict(torch.load('best_lstm'))
#    optimizer = torch.optim.Adam(rnn.parameters(),lr = 0.001)
#    loss_func = nn.MSELoss()
#    best_eval_arg_loss=10
    numpy_data_list=[]
    with torch.no_grad():
        rnn.eval()
        for j,batch in enumerate(inferece_test):
            data=torch.from_numpy(batch[0:5,:,:].reshape(1,392,5)).float().cuda()
            
#            print(data.shape)
            for day in range(30):
                print(data.reshape(5,-1))
                pred=rnn(data)
                pred=pred.reshape(1,-1)
#                tmp=data.clone()
                data=data.reshape(5,-1)
                
                
                data[:4,:]=data[1:5,:]
                data[4,:]=pred
#                pred=pred.reshape(1,392)
                numpy_data_list.append(pred.cpu().numpy())
#                print(data.reshape(5,-1))
                data=data.reshape(1,392,5)
#                break
        numpy_data=np.array(numpy_data_list).reshape(30,392)
        np.savetxt("region_migration_30.txt", numpy_data,fmt='%f',delimiter=',')
#        print(numpy_data.shape)
                
#        data=torch.from_numpy(batch[:,0:5,:,:].reshape(1,392,5)).float().cuda()
#        label=torch.from_numpy(batch[:,5,:,:].reshape(1,392,1)).float().cuda()
#        output=rnn(data)
#        loss=loss_func(output,label)
#        optimizer.zero_grad()
#        loss.backward()
#        totol_loss+=loss.item()
#        optimizer.step()
##            if j%10==0:
##                print("Epoch{} || loss{}".format(i,loss.item()))
#    print("Epoch:  {} || train_loss:   {}".format(i,totol_loss/len(train_loader)))
#    if i%5==0:
#        eval_totol_loss=0
#        with torch.no_grad():
#            rnn.eval()
#            for k,batch in enumerate(eval_loader):
#                eval_data=torch.from_numpy(batch[:,0:5,:,:].reshape(1,392,5)).float().cuda()
#                eval_label=torch.from_numpy(batch[:,5,:,:].reshape(1,392,1)).float().cuda()
#                eval_output=rnn(eval_data)
#                eval_loss=loss_func(eval_output,eval_label)
#                eval_totol_loss+=eval_loss.item()
#            eval_totol_arg_loss=eval_totol_loss/len(eval_loader)
#            if eval_totol_arg_loss<best_eval_arg_loss:
#                torch.save(rnn.state_dict(),'best_lstm'.format(i,eval_totol_arg_loss))
#                best_eval_arg_loss=eval_totol_arg_loss
#                
#            print("Epoch: {} || eval_loss:  {}".format(i,eval_totol_arg_loss))
#        rnn.train()


