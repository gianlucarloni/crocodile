import numpy as np
import torch
from torch import nn
from torch.nn import init
from .ACmixAttention import ACmix

class PositionAttentionModule(nn.Module):

    def __init__(self,d_model=512,kernel_size=3,H=7,W=7):
        super().__init__()
        self.cnn=nn.Conv2d(d_model,d_model,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.acmix = ACmix(in_planes=d_model, out_planes=d_model)

    def forward(self,x):
        bs,c,h,w=x.shape
        y=self.cnn(x)
        y = self.acmix(y).flatten(2).permute(0,2,1)
        return y


class ChannelAttentionModule(nn.Module):

    def __init__(self,d_model=512,kernel_size=3,H=7,W=7):
        super().__init__()
        self.cnn=nn.Conv2d(d_model,d_model,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.acmix = ACmix(in_planes=d_model, out_planes=d_model)

    def forward(self,x):
        bs,c,h,w=x.shape
        y=self.cnn(x)
        y = self.acmix(y).flatten(2)
        return y




class DModule(nn.Module):

    def __init__(self,d_model=2048,kernel_size=3,H=8,W=8):
        super().__init__()
        self.position_attention_module=PositionAttentionModule(d_model=d_model,kernel_size=3,H=8,W=8)
        self.channel_attention_module=ChannelAttentionModule(d_model=d_model,kernel_size=3,H=8,W=8)
    
    def forward(self,input):
        bs,c,h,w=input.shape
        p_out=self.position_attention_module(input)
        c_out=self.channel_attention_module(input)
        p_out=p_out.permute(0,2,1).view(bs,c,h,w)
        c_out=c_out.view(bs,c,h,w)
        return p_out+c_out

