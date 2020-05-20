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
    input_data=input_data.reshape(1,392)
    label_=label_.reshape(1,392)
    distance=torch.log(input_data+1)-torch.log(label_+1)
    # print(distance)
    distance_muti=torch.sum(distance.mul(distance))
    # print(distance_muti)
    result_rmsle=torch.sqrt(distance_muti/392)
    return result_rmsle






if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_num', type=int, default=392)
    parser.add_argument('--feat_dim', type=int, default=1)
    parser.add_argument('--n_his', type=int, default=5)
    parser.add_argument('--n_pred', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save', type=int, default=10)
    parser.add_argument('--Ks', type=int, default=1)  #equal to num_layers
    parser.add_argument('--Kt', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--keep_prob', type=float, default=0.5)
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
    
    
    
    
#    modeloutput_layer
    model=Model(args,T,n).cuda()
    
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            # nn.init.normal_(m.weight.data)
            # nn.init.xavier_normal_(m.weight.data)
            nn.init.kaiming_normal_(m.weight.data)#卷积层参数初始化
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            # m.weight.data.normal_()#全连接层参数初始化
            m.weight.data.normal_(0, 1)
            m.bias.data.zero_()


#    dataset
    dataset = InfectDataset(args)
    train, valid, test = data_split(dataset, args)
    train_loader = DataLoader(train, batch_size=1, shuffle=False)
    val_loader=DataLoader(valid, batch_size=1, shuffle=False)
    
#    opt
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
    # optimizer =  torch.optim.SGD(model.parameters(),lr = 0.0001, momentum = 0.9)
#    loss_function
    loss_func = nn.MSELoss(size_average = False)
    # loss_func=nn.L1Loss()
    # eval_rmsle=
    model.train()
    best_eval_arg_loss=1000
    for epoch in range(args.epochs):
        epoch_totol_loss_=0
        
        for i,batch in enumerate(train_loader):
            data=torch.from_numpy(batch[:,0:5,:,:]).reshape(-1,1,5,392).float().cuda()
            label=torch.from_numpy(batch[:,5,:,:]).reshape(-1,1,1,392).float().cuda()
            # print(label)
            
            out=model(data)
            # print(out.shape)
            # print(label.shape)

            # print(out*1000)
            optimizer.zero_grad()
            # loss=RMSLE(out,label)
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
                    eval_data=torch.from_numpy(batch_[:,0:5,:,:]).reshape(-1,1,5,392).float().cuda()
                    eval_label=torch.from_numpy(batch_[:,5,:,:]).reshape(-1,1,1,392).float().cuda()
                    eval_output=model(eval_data)[:,:,0,:]
                    # print(eval_output.shape)
                    # print(eval_output*1000)
                    # eval_output=(eval_output*1000)
                    # print(eval_output.shape)
                    eval_output[eval_output<0]=0
                    eval_loss=RMSLE(eval_output,eval_label)
                    # eval_loss=loss_func(eval_output*1000,eval_label*1000)
                    eval_totol_loss+=eval_loss.item()
                eval_totol_arg_loss=eval_totol_loss/len(val_loader)
                
                if eval_totol_arg_loss<best_eval_arg_loss:
                    torch.save(model.state_dict(),'best_model'.format(i,eval_totol_arg_loss))
                    best_eval_arg_loss=eval_totol_arg_loss
                    
                print("Epoch: {} || eval_loss:  {}".format(epoch,eval_totol_arg_loss))
            model.train()
            
            
            
            
        

