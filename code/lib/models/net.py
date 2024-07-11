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

class CausalityMapBlock(nn.Module):
    def __init__(self):
        '''
        Block to compute the causality maps of the deep neural network's latent features
        From papers:
            1- "Carloni, G., & Colantonio, S. (2024). Exploiting causality signals in medical images: A pilot study with empirical results. Expert Systems with Applications, 123433."
            
            2- "Carloni, G., Pachetti, E., & Colantonio, S. (2023). Causality-Driven One-Shot Learning for Prostate Cancer Grading from MRI. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 2616-2624)."
        
            Here, we adapt it to utilize the embedding of the transformer's output: Q (and potentially of Q_bar, Q_crocodile, Q_bar_crocodile, as a future extension of this work)
            To lower the computational burden, we leverage only the 'max' option instead of the 'lehmer' one (possible future extension of this work: utilize lehmer mean).
        '''
        super(CausalityMapBlock, self).__init__()
        
    def forward(self,x): #(bs,k,n,n)   #e.g. torch.Size([6, 9, 2048])
        
        if torch.isnan(x).any():
            print(f"...the current feature object contains NaN")
            raise ValueError
        
        # print(f"FORWARD Q: min={x.min()}, mean={x.mean()}, max={x.max()}")  
        if x.min() < 0: #TODO
            # Calculate the absolute minimum value for each batch
            min_values, _ = torch.min(x, dim=2)  # Only need the minimum values: Min along embedding dimension (dim=2)
            min_values = torch.abs(min_values)  # Take the absolute value
            # Unsqueeze to broadcast the minimum values across the class dimension
            min_values = min_values.unsqueeze(-1)
            # Add the minimum values to the tensor to achieve the shift
            x = x + min_values
        
        # print(f"FORWARD Q (shift): min={x.min()}, mean={x.mean()}, max={x.max()}")  

        maximum_values = torch.max(x, dim=2)[0]  # max: (bs,k) (6, 9)
        MAX_F = torch.max(maximum_values, dim=1)[0]  #MAX: (bs,) (6,) #get the global maximum of the features (strongest activation) for each image in the batch
        # print(f"FORWARD MAX_F: {MAX_F}")  

        x_div_max=x/(MAX_F.unsqueeze(1).unsqueeze(2) +1e-8) #TODO added epsilon; #implement batch-division: each element of each feature map gets divided by the respective MAX_F of that batch
        x = torch.nan_to_num(x_div_max, nan = 0.0)
        # print(f"FORWARD x (divmax): min={x.min()}, mean={x.mean()}, max={x.max()}")  

        #Note: to prevent ill posed divisions and operations, we sometimes add small epsilon (e.g., 1e-8) and nan_to_num() command.
        sum_values = torch.sum(x, dim=2) # sum: (bs,k) (6, 9)
        if torch.sum(torch.isnan(sum_values))>0:
            sum_values = torch.nan_to_num(sum_values,nan=0.0)
        # print(f"FORWARD sum_values: min={sum_values.min()}, mean={sum_values.mean()}, max={sum_values.max()}")  

        maximum_values = torch.max(x, dim=2)[0]  # max: (bs,k) (6, 9)
        # print(f"FORWARD maximum_values: min={maximum_values.min()}, mean={maximum_values.mean()}, max={maximum_values.max()}")  

        mtrx = torch.einsum('bi,bj->bij',maximum_values,maximum_values) #batch-wise outer product, the max value of mtrx object is 1.0
        # print(f"FORWARD mtrx: min={mtrx.min()}, mean={mtrx.mean()}, max={mtrx.max()}")  

        tmp = mtrx/(sum_values.unsqueeze(1) +1e-8) #TODO added epsilon
        causality_maps = torch.nan_to_num(tmp, nan = 0.0)

        # print(f"causality_maps: min={causality_maps.min()}, mean={causality_maps.mean()}, max={causality_maps.max()}")  
        return causality_maps
###################


class CROCODILE(nn.Module):
    def __init__(self, backbone, transfomer, num_class,
                 backbone_crocodile=None, transformer_crocodile=None, num_class_crocodile=None,
                 bs=8, backbonename="resnet50", imgsize=448, dropoutrate=0.35,
                 useCrocodile=True, useContrastiveLoss=True, contrastiveLossSpace="representation", useCausalityMaps=True):

        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_class

        ## Potentially, the Domain/Dataset branch could assume a different backbone and transformer setting (default, if not None, is the same as the disease branch)
        self.useCrocodile = useCrocodile
        self.backbone_crocodile = backbone_crocodile
        self.transformer_crocodile = transformer_crocodile
        self.num_class_crocodile = num_class_crocodile

        self.useContrastiveLoss = useContrastiveLoss
        self.contrastiveLossSpace = contrastiveLossSpace

        self.useCausalityMaps = useCausalityMaps

        self.backbonename = backbonename
        self.imgsize = imgsize

        self.sigmoid = nn.Sigmoid()
        self.seq = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(2),
            nn.Conv1d(2048, 2048, kernel_size=1),#TODO 2: seconda versione con kernel 1x1
            nn.Sigmoid(),
        )

        self.bs = bs

        if self.backbonename in ["resnet101","resnet50"]:
            dowmsphw = int(self.imgsize/16) #eg, 512->32, 480->30, 256->16, 128->8, 64->4
        else:
            print(f"Net.py>Causal().init --- unrecognised imgsize and backbone combination, dowmsphw {dowmsphw} is likely to cause issues")
            dowmsphw = 26 #TODO its original value

        self.hw = dowmsphw
        hidden_dim = transfomer.d_model
        self.dim = hidden_dim
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(self.num_class, hidden_dim)       
        self.fc = GroupWiseLinear(self.num_class, hidden_dim, bias=True)
        self.fc_add = GroupWiseLinear1(self.num_class, hidden_dim, bias=True)         

        self.dnet=DModule(d_model=hidden_dim,kernel_size=3,H=dowmsphw,W=dowmsphw) #applies both Channel and Position attention, and sums them up
        self.conv = nn.Conv1d(2 * hidden_dim, hidden_dim,1)

        self.att = Attention(channel=2048) #TODO attention, it is still a hardcoded value

        #TODO
        self.drop = nn.Dropout(p=dropoutrate)
        self.fc_confounding = GroupWiseLinear(self.num_class, hidden_dim, bias=True)

        if self.useCrocodile:
            self.query_embed_crocodile = nn.Embedding(self.num_class_crocodile, hidden_dim)  
            self.fc_crocodile = GroupWiseLinear(self.num_class_crocodile, hidden_dim, bias=True)
            self.fc_add_crocodile = GroupWiseLinear1(self.num_class_crocodile, hidden_dim, bias=True) 
            self.fc_confounding_crocodile = GroupWiseLinear(self.num_class_crocodile, hidden_dim, bias=True)

        if self.useCausalityMaps:
            self.causality_map_extractor = CausalityMapBlock()

    def forward(self, input):
        if torch.isnan(input).any():
            print("FORWARD - input has NaNs")
            raise ValueError
        
        ###### CAU MED block: the disease prediction branch
        # print(f"input: {input.size()}") 
        src, pos = self.backbone(input) # the input xray images pass through the CNN backbone and we obtain the 'src' featuremaps, with 'pos' embedding
        src, pos = src[-1], pos[-1]
        # print(f"src (after backbone): {src.size()}")
        if torch.isnan(src).any() or torch.isnan(pos).any():
            print("FORWARD - SRC or POS has NaNs")
            raise ValueError        
        ## Feature Learning block####
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
        #### Causal Learning block of the Fig.3.####
        query_input = self.query_embed.weight #TODO query_input: torch.Size([15, 2048])
        # print(f"query_input: {query_input.size()}")        
        #To get two outputs of Transformer, we use modified nn.functional.multi_head_attention_forward
        # ToDo: Migrate source code changes(nn.functional.multi_head_attention_forward) to transformer.py
        Q = self.transformer(self.input_proj(src), query_input, pos)[0]  # B,K,d ([1, bs, 15, 2048]) ##These are the Q features in Fig.3.
        # print(f"FORWARD - Q: {Q.size()}") #e.g.:torch.Size([1, 6, 9, 2048]), batch size 6 and 9 radiological findings
        Q_bar = self.transformer(self.input_proj(src), query_input, pos)[1]  # B,K,d  ##These are the Q^_ (bar) features in Fig.3.
        # print(f"useless: {useless.size()}")
        Q = Q[-1]
        # print(f"FORWARD - Q=Q[-1]: {Q.size()}") #e.g. torch.Size([6, 9, 2048])
        Q_bar = Q_bar[-1]
        num = Q.shape[0] # e.g., 6, the batch size    
        # print(f"FORWARD - num = Q.shape[0]: {num}")   
        l = [i for i in range(num)]
        random.shuffle(l)
        # random_idx = torch.tensor(l)
        random_idx = torch.as_tensor(l) #TODO 20 oct
        # print(f"net.py>Causal>forward: Q.shape {Q.shape} --> num: {num}, l(shuffle): {l}, and random_idx: {random_idx}") #TODO        
        Q_randomAdd = self.drop(Q_bar[random_idx]) + Q #the confounding features get shuffled and then randomly zeroed out, and later added to the causal features
        z_c_cap = self.fc_add(Q_randomAdd) ##These should be the z_c^, or at least the the 'h_x + h_c^' (intervened backdoor)
        z_x = self.fc(Q) ##These should be the z_x=CLS(h_x), or at least the h_x = MLP_causal(Q) in Eq.3 in the paper
        # print(f"FORWARD - z_x = self.fc(Q): {z_x.size()}") #e.g., torch.Size([6, 9])
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
        
        if self.useCausalityMaps:
            cmap_Q = self.causality_map_extractor(Q)
        else:
            cmap_Q = None

        if (self.useCrocodile and self.backbone_crocodile is not None and self.num_class_crocodile is not None and self.transformer_crocodile is not None):
            ###### CROCODILE block: the disease prediction branch
            src_crocodile, pos_crocodile = self.backbone_crocodile(input) # the input xray images pass through the CNN backbone for the domain/dataset predicition and we obtain the 'src' featuremaps, with 'pos' embedding
            src_crocodile, pos_crocodile = src_crocodile[-1], pos_crocodile[-1]
            if torch.isnan(src_crocodile).any() or torch.isnan(pos_crocodile).any():
                print("FORWARD - SRC_crocodile or POS_crocodile has NaNs")
                raise ValueError        
            ## Feature Learning block####
            src0_crocodile = self.dnet(src_crocodile) # position- and channel- attention, summed up, to be merged (concatenated) with the other features
            src00_crocodile = self.seq(src_crocodile) #TODO secondo me questo si può anche levare, tanto c'è già la parte di self.att(src)
            src1_crocodile = src00_crocodile.flatten(2) + self.att(src_crocodile).flatten(2)
            src2_crocodile = torch.cat((src0_crocodile.flatten(2), src1_crocodile), 1)
            src_crocodile = self.conv(src2_crocodile).reshape(self.bs, self.dim, self.hw, self.hw)
            if torch.isnan(src_crocodile).any():
                print("FORWARD - src_crocodile (after feature learning block) has NaNs")
                raise ValueError        
            #### Causal Learning block####
            query_input_crocodile = self.query_embed_crocodile.weight #TODO For instance, with 4 source datasets query_input: torch.Size([4, 2048])            
            Q_crocodile = self.transformer_crocodile(self.input_proj(src_crocodile), query_input_crocodile, pos_crocodile)[0]  # B,K,d ([1, bs, 4, 2048]) ##These are the Q features.
            Q_bar_crocodile = self.transformer_crocodile(self.input_proj(src_crocodile), query_input_crocodile, pos_crocodile)[1]  # B,K,d  ##These are the Q^_ (bar) features.
            Q_crocodile = Q_crocodile[-1]
            Q_bar_crocodile = Q_bar_crocodile[-1]
            num_crocodile = Q_crocodile.shape[0]        
            l_crocodile = [i for i in range(num_crocodile)]
            random.shuffle(l_crocodile)
            # random_idx = torch.tensor(l)
            random_idx_crocodile = torch.as_tensor(l_crocodile)
            Q_randomAdd_crocodile = self.drop(Q_bar_crocodile[random_idx_crocodile]) + Q_crocodile #the confounding features get shuffled and then randomly zeroed out
            z_c_cap_crocodile = self.fc_add_crocodile(Q_randomAdd_crocodile) ##These should be the z_c^, or at least the the 'h_x + h_c^' (intervened backdoor)
            z_x_crocodile = self.fc_crocodile(Q_crocodile) ##These should be the z_x=CLS(h_x), or at least the h_x = MLP_causal(Q) in Eq.3 in the paper
            z_c_crocodile = self.fc_confounding_crocodile(Q_bar_crocodile) #This is the MLP_confounding        
            if torch.isnan(z_c_cap_crocodile).any():
                print("FORWARD - z_c_cap_crocodile (just before returning) has NaNs")
                # raise ValueError
            if torch.isnan(z_x_crocodile).any():
                print("FORWARD - z_x_crocodile (just before returning) has NaNs")
                # raise ValueError
            if torch.isnan(z_c_crocodile).any():
                print("FORWARD - z_c_crocodile (just before returning) has NaNs")
                # raise ValueError
        else:
            z_x_crocodile = z_c_cap_crocodile = z_c_crocodile = Q_crocodile = Q_bar_crocodile=None

        # if self.useCrocodile:
        #     if self.useContrastiveLoss:  # caumed block + domain/dataset block + contrastiveLosses                      
        #         if self.contrastiveLossSpace == "activation":
        #             return z_x, z_c_cap, z_c, z_x_crocodile, z_c_cap_crocodile, z_c_crocodile
        #         elif self.contrastiveLossSpace == "representation":
        #             if self.useCausalityMaps:
        #                 return z_x, z_c_cap, z_c, z_x_crocodile, z_c_cap_crocodile, z_c_crocodile, Q, Q_bar, Q_crocodile, Q_bar_crocodile, cmap_Q 
        #             else:
        #                 return z_x, z_c_cap, z_c, z_x_crocodile, z_c_cap_crocodile, z_c_crocodile, Q, Q_bar, Q_crocodile, Q_bar_crocodile 
        #     else: # caumed block + domain/dataset block
        #         return z_x, z_c_cap, z_c, z_x_crocodile, z_c_cap_crocodile, z_c_crocodile
        # else: #just the caumed block
        #     return z_x, z_c_cap, z_c
        ## TODO:
        return z_x, z_c_cap, z_c, z_x_crocodile, z_c_cap_crocodile, z_c_crocodile, Q, Q_bar, Q_crocodile, Q_bar_crocodile, cmap_Q

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

    logger.info("Creating the backbone for the CauMed branch (disease classification)...")
    backbone = build_backbone(args,logger=logger)
    backbone_crocodile = None #initialize to none, for the time being
    
    logger.info("Creating the transformer for the CauMed branch (disease classification)...")
    transformer = build_transformer(args)
    transformer_crocodile = None #initialize to none, for the time being

    if args.useCrocodile:
        logger.info("With args.useCrocodile==True, we need to add the additional Dataset/Domain branch to the model architecture:")
        logger.info("Creating the backbone for the Dataset/Domain branch (environment classification)...")
        backbone_crocodile = build_backbone(args,logger=logger)
        logger.info("Creating the transformer for the Dataset/Domain branch (environment classification)...")
        transformer_crocodile = build_transformer(args)

    model = CROCODILE(
        backbone=backbone,
        transfomer=transformer,
        num_class=args.num_class,
        backbone_crocodile=backbone_crocodile, ##TODO
        transformer_crocodile=transformer_crocodile, ##TODO
        num_class_crocodile=args.num_class_crocodile, ##TODO
        bs = int(args.batch_size / dist.get_world_size()),
        backbonename=args.backbone,
        imgsize=args.img_size,
        dropoutrate=args.dropoutrate_randomaddition, #TODO added
        useCrocodile=args.useCrocodile, #TODO added
        useContrastiveLoss=args.useContrastiveLoss, #TODO added
        contrastiveLossSpace=args.contrastiveLossSpace, #TODO added
        useCausalityMaps=args.useCausalityMaps
    )

    logger.info("|\t Model created.")

    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("Traditional CauMed: Set model.input_proj to Indentity()! Indeed, this keeps the input projection layer, and this is needed when the channel of image features is different from hidden_dim of Transformer layers.")

    
    return model