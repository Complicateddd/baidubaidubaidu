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




def RMSLE(input_data,label_):
    input_data=input_data.reshape(5,392)
    label_=label_.reshape(5,392)
    distance=torch.log(input_data+1)-torch.log(label_+1)
    # print(distance)
    distance_muti=torch.sum(distance.mul(distance))
    # print(distance_muti)
    result_rmsle=torch.sqrt(distance_muti/392*5)
    return result_rmsle







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
    
    parser.add_argument('--region_names_file', type=str, 
            default='/home/ubuntu/baidu/data_processed/region_names.txt')
    args = parser.parse_args()
    
    blocks = [[1, 32, 64], [64, 32, 128]]
    args.blocks = blocks
    a=torch.randn(1,1,5,392)
    _,channel,T,n=a.shape
    
    
    
    
#    model
    model=Model(args,T,n).cuda()
    
#    dataset
    dataset = InfectDataset(args)
    train, valid, test = data_split(dataset, args)
    train_loader = DataLoader(train, batch_size=1, shuffle=False)
    val_loader=DataLoader(valid, batch_size=1, shuffle=False)
    
#    opt
    # optimizer = torch.optim.Adam(model.parameters(),lr = 0.00001)
    optimizer =  torch.optim.SGD(model.parameters(),lr = 0.0001, momentum = 0.9)
#    loss_function
    loss_func = nn.MSELoss()
    # eval_rmsle=
    
    best_eval_arg_loss=100
    for epoch in range(args.epochs):
        epoch_totol_loss_=0
        
        for i,batch in enumerate(train_loader):
            data=torch.from_numpy(batch[:,0:5,:,:]).reshape(1,1,5,392).float().cuda()
            label=torch.from_numpy(batch[:,5:,:,:]).reshape(1,1,5,392).float().cuda()
            # print(label)
            out=model(data)
            # print(out)
            optimizer.zero_grad()
            loss=loss_func(out,label)
            loss.backward()
            optimizer.step()
            epoch_totol_loss_+=loss.item()
#            print(loss)
        print("Epoch:  {} || train_loss:   {}".format(epoch,epoch_totol_loss_/len(train_loader)))
        
        if epoch%5==0:
            eval_totol_loss=0
            with torch.no_grad():
                model.eval()
                for k,batch_ in enumerate(val_loader):
                    eval_data=torch.from_numpy(batch_[:,0:5,:,:]).reshape(1,1,5,392).float().cuda()
                    eval_label=torch.from_numpy(batch_[:,5:,:,:]*1000).reshape(1,1,5,392).float().cuda()
                    eval_output=model(eval_data)
                    eval_output=(eval_output*1000)
                    # print(eval_output.shape)
                    
                    eval_loss=RMSLE(eval_output,eval_label)
                    eval_totol_loss+=eval_loss.item()
                eval_totol_arg_loss=eval_totol_loss/len(val_loader)
                
                if eval_totol_arg_loss<best_eval_arg_loss:
                    torch.save(model.state_dict(),'best_model'.format(i,eval_totol_arg_loss))
                    best_eval_arg_loss=eval_totol_arg_loss
                    
                print("Epoch: {} || eval_loss:  {}".format(epoch,eval_totol_arg_loss))
            model.train()
            
            
            
            
        

