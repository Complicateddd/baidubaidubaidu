# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:36:03 2020

@author: Administrator
"""
import torch
#from torch.autograd import Variable
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self,lstm_num,hitten_num,input_channel,output_channel):
        super(RNN,self).__init__() #面向对象中的继承
        self.lstm = nn.LSTM(input_channel,hitten_num,lstm_num) #输入数据2个特征维度，6个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果
        
        self.out = nn.Linear(hitten_num,output_channel) #线性拟合，接收数据的维度为6，输出数据的维度为1
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        x1,_ = self.lstm(x)
#        print(x1.shape)
        a,b,c = x1.shape
        out = self.out(x1.reshape(-1,c)) #因为线性层输入的是个二维数据，所以此处应该将lstm输出的三维数据x1调整成二维数据，最后的特征维度不能变
        # out=self.relu(out)
        out1 = out.reshape(a,b,-1) #因为是循环神经网络，最后的时候要把二维的out调整成三维数据，下一次循环使用
        return out1

#rnn = RNN()

if __name__=="__main__":
    a=torch.randn(1,392,5)
    rnn=RNN(3,6,5,1)
    out = rnn(a)
#    print(a)
    print(out.shape)