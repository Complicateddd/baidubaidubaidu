# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:30:41 2020

@author: Administrator
"""

import numpy as np
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:34:09 2020

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
from model import Model
import argparse
from Dataset import InfectDataset,data_split
from dataloader import DataLoader
import pandas as pd
import os

def save_to_submit(predicts, args):
    # (n_pred, 1, city_num, 1) --> (n_pred, city_num)
    predicts = np.squeeze(predicts)
    predicts = predicts * 1000
    predicts = predicts.transpose(1,0).reshape(-1,).int()
    #  log.info(predicts)
    predicts = pd.DataFrame({"ret": predicts})

    submit = pd.read_csv(args.submit_file, 
                          header=None,
                          names=["cityid", "regionid", "date", "cnt"])

    submit = pd.concat([submit, predicts], axis=1)
    submit = submit.drop(columns=['cnt'])

    submit.to_csv(os.path.join(args.output_path, 'submission.csv'), 
            index=False, header=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_num', type=int, default=392)
    parser.add_argument('--feat_dim', type=int, default=1)
    parser.add_argument('--n_his', type=int, default=5)
    parser.add_argument('--n_pred', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save', type=int, default=10)
    parser.add_argument('--Ks', type=int, default=1)  #equal to num_layers
    parser.add_argument('--Kt', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.005)
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
    parser.add_argument('--submit_file', type=str, 
            default='submission.csv')

    parser.add_argument('--region_names_file', type=str, 
            default='/home/ubuntu/baidu/data_processed/region_names.txt')
    args = parser.parse_args()
    
    blocks = [[1, 32, 64], [64, 32, 128]]
    args.blocks = blocks
    a=torch.randn(1,1,5,392)
    _,channel,T,n=a.shape
    
    
    
    
#    model
    model=Model(args,T,n).cuda()
    model.load_state_dict(torch.load('best_model'))
    model.eval()
#    dataset
    infer_data=np.loadtxt("region_migration_30.txt",delimiter=",")

    temp_list=[]
    for number in range(6):
        infer_batch_data=infer_data[number*5:(number+1)*5,:]

        data=torch.from_numpy(infer_batch_data).reshape(1,1,5,392).float().cuda()
        out=model(data)
        pred=out.reshape(5,1,392,1)
        temp_list.append(pred)
    result=torch.stack(temp_list)
    result=result.reshape(30,1,392,1)
    print(result.shape)
    save_to_submit(result.cpu(), args)

    
            
            
        

