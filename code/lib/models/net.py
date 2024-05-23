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


class CROCODILE(nn.Module):
    def __init__(self, backbone, transfomer, num_class,
                 backbone_crocodile=None, transformer_crocodile=None, num_class_crocodile=None,
                 bs=8, backbonename="resnet50", imgsize=448, dropoutrate=0.35,
                 useCrocodile=True, useContrastiveLoss=True, contrastiveLossSpace="representation"):

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
        ##

        self.backbonename = backbonename
        self.imgsize = imgsize

        # ekernel_size = 3
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

        # TODO 8 May 2024
        if self.useCrocodile:
            self.query_embed_crocodile = nn.Embedding(self.num_class_crocodile, hidden_dim)  
            self.fc_crocodile = GroupWiseLinear(self.num_class_crocodile, hidden_dim, bias=True)
            self.fc_add_crocodile = GroupWiseLinear1(self.num_class_crocodile, hidden_dim, bias=True) 
            self.fc_confounding_crocodile = GroupWiseLinear(self.num_class_crocodile, hidden_dim, bias=True)

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
        ## Feature Learning block of the Fig.3.####
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

        if self.useCrocodile:
            if self.useContrastiveLoss:  # caumed block + domain/dataset block + contrastiveLosses                      
                if self.contrastiveLossSpace == "activation":
                    return z_x, z_c_cap, z_c, z_x_crocodile, z_c_cap_crocodile, z_c_crocodile
                elif self.contrastiveLossSpace == "representation":
                    return z_x, z_c_cap, z_c, z_x_crocodile, z_c_cap_crocodile, z_c_crocodile, Q, Q_bar, Q_crocodile, Q_bar_crocodile 
            else: # caumed block + domain/dataset block
                return z_x, z_c_cap, z_c, z_x_crocodile, z_c_cap_crocodile, z_c_crocodile
        else: #just the caumed block
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
    )

    logger.info("|\t Model created.")

    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("Traditional CauMed: Set model.input_proj to Indentity()! Indeed, this keeps the input projection layer, and this is needed when the channel of image features is different from hidden_dim of Transformer layers.")

    
    return model












## TODO May 2024:############################################

# def build_net_DomainClassifier(args,logger=None):
#     logger.info("DomainClassifier: Creating the backbone")
#     backbone = build_backbone(args,logger=logger)

#     logger.info("DomainClassifier:Creating the transformer")
#     transformer = build_transformer(args)

#     model = DomainClassifier(
#         backbone=backbone,
#         transfomer=transformer,
#         num_class=args.DC_num_class, #the number of different datasets/domains that act as source training environments
#         bs = int(args.batch_size / dist.get_world_size()),
#         backbonename=args.backbone,
#         imgsize=args.img_size,
#         dropoutrate=args.dropoutrate_randomaddition, #TODO Anyway, we could remove it: for the task of domain/dataset classification,  we could not implement the backdoor adjustment and just consider the _sp and _c features
#     )

#     logger.info("Created the DomainClassifier (backbone+transformer)")

#     if not args.keep_input_proj:
#         model.input_proj = nn.Identity()
#         print("Set model.input_proj to Indentity()! Indeed, this keeps the input projection layer, and this is needed when the channel of image features is different from hidden_dim of Transformer layers.")

#     return model



























class FullyConnectedClassifier(nn.Module):
  """
  A simple fully-connected classifier class.
  """
  def __init__(self, in_features, hidden_layers, out_features, activation=nn.ReLU):
    """
    Args:
      in_features: Number of input features.
      hidden_layers: List of hidden layer sizes (number of neurons in each layer).
      out_features: Number of output features (number of classes).
      activation: Activation function to use between layers (defaults to ReLU).
    """
    super(FullyConnectedClassifier, self).__init__()
    layers = []
    # Input layer
    layers.append(nn.Linear(in_features, hidden_layers[0]))
    layers.append(activation())
    # Hidden layers
    for i in range(1, len(hidden_layers)):
      layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
      layers.append(activation())
    # Output layer
    layers.append(nn.Linear(hidden_layers[-1], out_features))

    # Sequential construction of the layers
    self.classifier = nn.Sequential(*layers)

  def forward(self, x):
    """
    Forward pass of the classifier.
    Args:
      x: Input tensor.
    Returns:
      Output logits of the classifier.
    """
    return self.classifier(x)

def build_domain_classifier(args, logger=None):
  """
  Builder function to create a FullyConnectedClassifier instance.
  Returns:
    An instance of FullyConnectedClassifier.
  """
  in_features = args.DC_in_features # Number of input features.
  hidden_layers = args.DC_hidden_layers # List of hidden layer sizes (number of neurons in each layer)
  out_features = args.DC_out_features # Number of output features (number of classes)
  activation = args.DC_activation #Activation function to use between layers (defaults to ReLU)

  logger.info("Building Domain Classifier model (fully connected)")

  return FullyConnectedClassifier(in_features, hidden_layers, out_features, activation)

def build_domain_classifier(args, logger=None):
    logger.info("Creating the domain/dataset classifier")
    model = Classifier(

    )