import torch
import torch.nn as nn
import torch.nn.functional as f



class same_con2d(nn.Module):
	def __init__(self,input_channel,output_channel,k_size=[4,1],stride=[1,1],bias=False):
		super(same_con2d, self).__init__()
		self.k_size=k_size
		self.stride=stride
		self.bias=bias
		self.sourse_conv2d=nn.Conv2d(input_channel,output_channel,k_size,stride,bias=bias)
	def forward(self,x):
		##此处假定stride默认为1,1的情况padding
		up_bottom_pad_num=(self.k_size[0]-1)//self.stride[0]
		lf_rg_pad_num=(self.k_size[1]-1)//self.stride[1]
		pad=nn.ZeroPad2d(padding=(lf_rg_pad_num,lf_rg_pad_num,up_bottom_pad_num//2,up_bottom_pad_num-up_bottom_pad_num//2))
		return self.sourse_conv2d(pad(x))


class Model(nn.Module):
    def __init__(self,args,T,n):
        super(Model, self).__init__()
        self.args=args
        Ko=self.args.n_his
        self.sq=nn.Sequential()
        self.sq.add_module('st1',st_conv_block(self.args.Ks,self.args.Kt,
                                               self.args.blocks[0],self.args.keep_prob,act_fun='GLU'))
        self.sq.add_module('st2',st_conv_block(self.args.Ks,self.args.Kt,
                                               self.args.blocks[1],self.args.keep_prob,act_fun='GLU'))
        
        self.sq.add_module('output',output_layer(Ko,n,128))
        # self.sq.add_module('R',nn.ReLU(inplace=True))
        
    def forward(self,x):
        out=self.sq(x)
        return out
        

class st_conv_block(nn.Module):
    def __init__(self,ks,kt,channels,keep_prob,act_fun='GLU'):
        super(st_conv_block, self).__init__()
        c_si, c_t, c_oo = channels
        self.temporal_conv_layer1=temporal_conv_layer(kt,c_si,c_t,act_function='GLU')
        self.spatio_conv_layer=spatio_conv_layer(ks,c_t,c_t)
        self.temporal_conv_layer2=temporal_conv_layer(kt,c_t,c_oo)
        self.batch_norm=nn.BatchNorm2d(c_oo)
        self.droupout=nn.Dropout(1.0-keep_prob)
        
    def forward(self,x):
        x=self.temporal_conv_layer1(x)
        x=self.spatio_conv_layer(x)
        x=self.temporal_conv_layer2(x)
        x=self.batch_norm(x)
        return self.droupout(x)
    
     
class temporal_conv_layer(nn.Module):
    def __init__(self,kt,c_in,c_out,act_function='relu'):
        super(temporal_conv_layer, self).__init__()
        self.kt=kt
        self.c_in=c_in
        self.c_out=c_out
        self.act_fun=act_function
        
        if self.c_in>self.c_out:
            self.x_input_conv=nn.Conv2d(c_in,c_out,[1,1],[1,1],bias=False)
        else:
            self.x_input_conv=None
            
        if act_function=='GLU':
            self.conv_1=same_con2d(c_out,c_out*2,[self.kt,1],[1,1],bias=True)
        else:
            self.Rule=nn.ReLU(inplace=True)
            self.conv_1=same_con2d(c_out,c_out,[self.kt,1],[1,1],bias=True)
        
    def forward(self,x):
        if self.c_in>self.c_out:
            x_input=self.x_input_conv(x)
        elif self.c_in<self.c_out:
            fill=torch.zeros(x.shape[0],self.c_out-self.c_in,x.shape[2],x.shape[3]).cuda()
            x_input=torch.cat((x,fill),1)
        else:
            x_input=x
        
        x_conv=self.conv_1(x_input)
        
        if self.act_fun=='GLU':
            
            return (x_conv[:,0:self.c_out,:,:] + x_input)*torch.sigmoid(x_conv[:,-self.c_out:,:,:])
        else: 
            if self.act_fun == "linear":
                return x_conv
            elif self.act_fun == "sigmoid":
                return torch.sigmoid(x_conv)
            elif self.act_fun == "relu":
                return self.Rule(x_conv + x_input)
            else:
                raise ValueError(
                    f'ERROR: activation function "{act_func}" is not defined.')
        
class spatio_conv_layer(nn.Module):
    def __init__(self,ks,c_in,c_out):
        super(spatio_conv_layer, self).__init__()
        self.ks=ks
        self.c_in=c_in
        self.c_out=c_out
        if self.c_in>self.c_out:
            self.x_input_conv=nn.Conv2d(c_in,c_out,[1,1],[1,1],bias=False)
        else:
            self.x_input_conv=None
        
        self.fc_module_list=nn.ModuleList([nn.Linear(self.c_out,self.c_out,bias=True) for i in range(self.ks)])
        self.re_module_list=nn.ModuleList([nn.ReLU(inplace=True) for i in range(self.ks)])
    def forward(self,x):
        _,_,h,w = x.shape
        if self.c_in>self.c_out:
            x_input=self.x_x_input_conv(x)
        elif self.c_in<self.c_out:
            fill=torch.zeros(x.shape[0],self.c_out-self.c_in,x.shape[2],x.shape[3])
            x_input=torch.cat((x,fill),1)
        else:
            x_input=x
        
        x_input = x_input.reshape(-1,self.c_out)
        for i in range(self.ks):
        
           x_input=self.fc_module_list[i](x_input)
           x_input=self.re_module_list[i](x_input)
        x_input=x_input.reshape(-1,self.c_out,h,w)
        return x_input

class output_layer(nn.Module):
    def __init__(self,T,n,channel):
        super(output_layer,self).__init__()
        self.temporal_conv_layer1=temporal_conv_layer(T,channel,channel,'GLU')
        self.norm=nn.BatchNorm2d(channel)
        self.temporal_conv_layer2=temporal_conv_layer(1,channel,channel,'sigmoid')
        self.fc=fully_con_layer(n,channel)
        
    def forward(self,x):
        out=self.temporal_conv_layer1(x)
#        print(out.shape)
        out=self.norm(out)
        out=self.temporal_conv_layer2(out)
        out=self.fc(out)
        return out
        

class fully_con_layer(nn.Module):
    def __init__(self,n,channel):
        super(fully_con_layer,self).__init__()
        self.fc_conv=same_con2d(channel,1,[1,1],[1,1],bias=True)
    def forward(self,x):
        return self.fc_conv(x)


        
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_num', type=int, default=392)
    parser.add_argument('--feat_dim', type=int, default=1)
    parser.add_argument('--n_his', type=int, default=30)
    parser.add_argument('--n_pred', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save', type=int, default=10)
    parser.add_argument('--Ks', type=int, default=1)  #equal to num_layers
    parser.add_argument('--Kt', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--keep_prob', type=float, default=1.0)
    args = parser.parse_args()
    
    blocks = [[1, 32, 64], [64, 32, 128]]
    args.blocks = blocks
    a=torch.randn(1,1,10,392)
    _,channel,T,n=a.shape
    model=Model(args,T,n)
    out=model(a)
    print(out.shape)
    
#    print(spatio_conv_layer1(temporal_conv_layer1(a)).shape)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    