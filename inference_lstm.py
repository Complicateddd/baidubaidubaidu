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
    parser.add_argument('--train_mode', type=str, default='X')


    parser.add_argument('--region_names_file', type=str, 
            default='/home/ubuntu/baidu/data_processed/region_names.txt')
    args = parser.parse_args()
    dataset = InfectDataset(args,args.train_mode)

    inferece_test=inferece_data_split(dataset,args)
#    train_loader = DataLoader(train, batch_size=1, shuffle=False)
#    eval_loader=DataLoader(valid, batch_size=1, shuffle=False)
    inferece_loader=DataLoader(inferece_test, batch_size=1, shuffle=False)
    
    rnn=RNN(3,128,args.n_his,1).cuda()
    rnn.load_state_dict(torch.load('feature_best_lstm.pth'))
#    optimizer = torch.optim.Adam(rnn.parameters(),lr = 0.001)
#    loss_func = nn.MSELoss()
#    best_eval_arg_loss=10
    numpy_data_list=[]
    with torch.no_grad():
        rnn.eval()
        for j,batch in enumerate(inferece_test):
            data=torch.from_numpy(batch[0:5,:,:].reshape(1,392,5)).float().cuda()
            
#            print(data.shape)
            for day in range(50):
                # print(data.reshape(5,-1))
                pred=rnn(data)
                pred=pred.reshape(1,-1)
                pred[pred<0]=0.
                print(pred)
#                tmp=data.clone()
                data=data.reshape(5,-1)
                
                
                data[:4,:]=data[1:5,:]
                data[4,:]=pred
#                pred=pred.reshape(1,392)
                numpy_data_list.append(pred.cpu().numpy()*1000)
#                print(data.reshape(5,-1))
                data=data.reshape(1,392,5)
#                break
        numpy_data=np.array(numpy_data_list).reshape(50,392)
        np.savetxt("label_migration_30.txt", numpy_data,fmt='%f',delimiter=',')
#        


