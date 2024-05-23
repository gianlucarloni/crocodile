import os, sys
import os.path as osp
import random
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math
from collections import OrderedDict
from models.backbone import build_backbone
from models.transformer import build_transformer
from models.DNet import DModule

def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict

class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class GroupWiseLinear1(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class GroupWiseLinear2(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class Attention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        b, c, _, _ = x.size()
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        y = self.relu(x1 + x2 + x3)
        return y


class Causal(nn.Module):
    def __init__(self, backbone, transfomer, num_class, bs, backbonename="resnet101", imgsize=224, dropoutrate=0.35):

        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_class

        self.backbonename = backbonename
        self.imgsize = imgsize

        # ekernel_size = 3
        self.sigmoid = nn.Sigmoid()
        self.seq = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # ####
            # nn.Conv1d(1, 1, kernel_size=ekernel_size, padding=(ekernel_size - 1) // 2),
            # nn.Sigmoid(),
            # ###

            ##TODO mio:
            nn.Flatten(2),
            # nn.Conv1d(2048, 2048, kernel_size=3, padding=1),
            nn.Conv1d(2048, 2048, kernel_size=1),#TODO 2: seconda versione con kernel 1x1
            nn.Sigmoid(),

        )

        self.bs = bs

        if self.backbonename in ["resnet101","resnet50"]:
            dowmsphw = int(self.imgsize/16) #eg, 512->32, 480->30, 256->16, 128->8
        else:
            print(f"Net.py>Causal().init --- unrecognised imgsize and backbone combination, dowmsphw {dowmsphw} is likely to cause issues")
            dowmsphw = 26 #TODO its original value

        self.hw = dowmsphw
        hidden_dim = transfomer.d_model
        self.dim = hidden_dim
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        # self.fc = torch.jit.script(GroupWiseLinear(num_class, hidden_dim, bias=True))
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)

        # self.fc_add = torch.jit.script(GroupWiseLinear1(num_class, hidden_dim, bias=True))
        self.fc_add = GroupWiseLinear1(num_class, hidden_dim, bias=True)

        # self.fc_cat = torch.jit.script(GroupWiseLinear2(num_class, hidden_dim*2, bias=True))
        self.fc_cat = GroupWiseLinear2(num_class, hidden_dim*2, bias=True)

        self.dnet=DModule(d_model=hidden_dim,kernel_size=3,H=dowmsphw,W=dowmsphw) #applies both Channel and Position attention, and sums them up
        self.conv = nn.Conv1d(2 * hidden_dim, hidden_dim,1)

        # self.att = torch.jit.script(Attention(channel=2048))
        self.att = Attention(channel=2048)

        self.cat_or_add = "cat"

        #TODO
        self.drop = nn.Dropout(p=dropoutrate)
        self.fc_confounding = GroupWiseLinear(num_class, hidden_dim, bias=True)

    def forward(self, input):
        if torch.isnan(input).any():
            print("FORWARD - input has NaNs")
            raise ValueError
        
        # print(f"input: {input.size()}") 
        src, pos = self.backbone(input) # the input xray images pass through the CNN backbone and we obtain the 'src' featuremaps, with 'pos' embedding
        src, pos = src[-1], pos[-1]
        # print(f"src (after backbone): {src.size()}")
        if torch.isnan(src).any() or torch.isnan(pos).any():
            print("FORWARD - SRC or POS has NaNs")
            raise ValueError
        
        #### Feature Learning block of the Fig.3.####

        src0 = self.dnet(src) # position- and channel- attention, summed up, to be merged (concatenated) with the other features
        # print(f"src0 (after dnet): {src0.size()}")

        src00 = self.seq(src) #TODO secondo me questo si può anche levare, tanto c'è già la parte di self.att(src)
        # print(f"src00 (after seq): {src00.size()}")
        src1 = src00.flatten(2) + self.att(src).flatten(2)
        # print(f"src1 (after sum): {src1.size()}")
        
        src2 = torch.cat((src0.flatten(2), src1), 1)
        # print(f"src2 (after concat): {src2.size()}")

        # src = self.conv(src2).reshape(self.bs, self.dim, self.hw, self.hw)
        src = self.conv(src2).reshape(self.bs, self.dim, self.hw, self.hw)
        # print(f"src (after final conv pre-transformer): {src.size()}")
        if torch.isnan(src).any():
            print("FORWARD - src (after feature learning block) has NaNs")
            raise ValueError
        ########




        #### Causal Learning block of the Fig.3.####
        query_input = self.query_embed.weight #TODO query_input: torch.Size([15, 2048])
        # print(f"query_input: {query_input.size()}")
        
        #To get two outputs of Transformer, we use modified nn.functional.multi_head_attention_forward
        # ToDo: Migrate source code changes(nn.functional.multi_head_attention_forward) to transformer.py
        Q = self.transformer(self.input_proj(src), query_input, pos)[0]  # B,K,d ([1, bs, 15, 2048]) ##These are the Q features in Fig.3.
        # print(f"hs: {hs.size()}")
        Q_bar = self.transformer(self.input_proj(src), query_input, pos)[1]  # B,K,d  ##These are the Q^_ (bar) features in Fig.3.
        # print(f"useless: {useless.size()}")

        Q = Q[-1]
        Q_bar = Q_bar[-1]

        num = Q.shape[0]        
        l = [i for i in range(num)]
        random.shuffle(l)
        # random_idx = torch.tensor(l)
        random_idx = torch.as_tensor(l) #TODO 20 oct

        # print(f"net.py>Causal>forward: Q.shape {Q.shape} --> num: {num}, l(shuffle): {l}, and random_idx: {random_idx}") #TODO
        
        #TODO in questa versione di loro, sembra che -o con cat o con add- le usa tutte le feature useless
        # invece, nel paper dichiarano che usano una random addition al 30-40%, quindi non tutte.
        # Potrei quindi azzerare il 70-60% di queste features risultanti, cosi che la dimensionalità rimanga
        # la stessa ma in quelle posizioni non ci sono valori, di conseguenza devo aggiornare anche il fully connected in accordo
        ## Inoltre, non c'è per niente l'uscita delle sole features confounder, ovvero h_c (o z_c),
        #    che andrebbe inserita per far ritornare anche quei logit
        
        # if self.cat_or_add == "cat":
        #     halfuse = torch.cat((useless[random_idx], hs), dim=2)
        #     halfout = self.fc_cat(halfuse)
        # else:
        #     halfuse = useless[random_idx] + hs
        #     halfout = self.fc_add(halfuse) ##These should be the z_c^, or at least the the 'h_x + h_c^' (intervened backdoor)

        # out = self.fc(hs) ##These should be the z_x=CLS(h_x), or at least the h_x = MLP_causal(Q) in Eq.3 in the paper

        # # import ipdb; ipdb.set_trace()
        # return out, halfout  

        ## TODO infatti ecco la mia: solo ADD, senza CAT, e non il 100% ma una frazione

        Q_randomAdd = self.drop(Q_bar[random_idx]) + Q #the confounding features get shuffled and then randomly zeroed out
        z_c_cap = self.fc_add(Q_randomAdd) ##These should be the z_c^, or at least the the 'h_x + h_c^' (intervened backdoor)
        z_x = self.fc(Q) ##These should be the z_x=CLS(h_x), or at least the h_x = MLP_causal(Q) in Eq.3 in the paper
        z_c = self.fc_confounding(Q_bar) #This is the MLP_confounding
        
        if torch.isnan(z_c_cap).any():
            print("FORWARD - z_c_cap (just before returning) has NaNs")
            # raise ValueError
        if torch.isnan(z_x).any():
            print("FORWARD - z_x (just before returning) has NaNs")
            # raise ValueError
        if torch.isnan(z_c).any():
            print("FORWARD - z_c (just before returning) has NaNs")
            # raise ValueError
        
        return z_x, z_c_cap, z_c 

    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(),
                     self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))

def build_net(args,logger=None):
    logger.info("Creo la backbone")
    backbone = build_backbone(args,logger=logger)

    logger.info("Creo il transformer")
    transformer = build_transformer(args)

    model = Causal(
        backbone=backbone,
        transfomer=transformer,
        num_class=args.num_class,
        bs = int(args.batch_size / dist.get_world_size()),
        # bs = int(args.batch_size / 1),#TODO
        backbonename=args.backbone,
        imgsize=args.img_size,
        dropoutrate=args.dropoutrate_randomaddition,
    )

    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("Set model.input_proj to Indentity()! Indeed, this keeps the input projection layer, and this is needed when the channel of image features is different from hidden_dim of Transformer layers.")

    return model
