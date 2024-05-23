from collections import OrderedDict
import os
import warnings

import torch
from torch.functional import Tensor
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import models
from .position_encoding import build_position_encoding

## TODO 20 oct
# mydir=os.path.join(os.getcwd(), 'pretrained_models')
from _init_paths import pretrainedmodels_path
mydir=pretrainedmodels_path
torch.hub.set_dir(mydir)
os.environ['TORCH_HOME']=mydir
##

def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict

def get_model_path(modelname):
    """
        Config your pretrained model path here!
    """
    # raise NotImplementedError("Please config your pretrained modelpath!")
    pretrained_dir = '/config/to/your/pretrained/model/dir/if/needed'
    PTDICT = {
        'CvT_w24': 'CvT-w24-384x384-IN-22k.pth',
        'swin_L_384_22k': 'swin_large_patch4_window12_384_22k.pth',
        'tresnetl': 'tresnet_l_448.pth',
        'tresnetxl': 'tresnet_xl_448.pth',
        'tresnetl_v2': 'tresnet_l_v2_miil_21k.pth',
    }
    return os.path.join(pretrained_dir, PTDICT[modelname]) 

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_layers: Dict,
    # return_interm_layers: bool
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, input: Tensor):
        xs = self.body(input)
        out: Dict[str, Tensor] = {}
        for name, x in xs.items():
            out[name] = x
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 pretrained: bool = True, logger=None): #TODO logger
        if name in ['resnet18', 'resnet50', 'resnet34', 'resnet101']:
            # backbone = getattr(torchvision.models, name)( #TODO May 2024 sembra che li importasse da torchvision invece io voglio pescare dallo script resnet.py che viene importato in init
            backbone = getattr(models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                # pretrained=True, #TODO:
                pretrained=pretrained,
                norm_layer=FrozenBatchNorm2d, logger=logger) #TODO logger
            if return_interm_layers:
                return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            else:
                return_layers = {'layer4': "0"}
        # elif name in ['tresnetl', 'tresnetxl', 'tresnetl_v2']:
        #     backbone = getattr(models, name)(
        #         {'num_classes': 1}
        #     )
        #     # load pretrained model
        #     if pretrained:
        #         pretrainedpath = get_model_path(name)
        #         # checkpoint = torch.load(pretrainedpath, map_location='cpu')
        #         checkpoint = torch.load(pretrainedpath, map_location='cuda') #TODO 20 oct
        #         from collections import OrderedDict
        #         if 'model' in checkpoint:
        #             checkpoint = checkpoint['model']
        #         if 'state_dict' in checkpoint:
        #             checkpoint = checkpoint['state_dict']
        #         _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if 'head.fc' not in k})
        #         backbone.load_state_dict(_tmp_st, strict=False)
            
        #     if return_interm_layers:
        #         raise NotImplementedError('return_interm_layers must be False in TResNet!')
        #     return_layers = {'body': "0"}
        else:
            raise NotImplementedError("Unknow name: %s" % name)
            
        NCDICT = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048,
            # 'tresnetl': 2432,
            # 'tresnetxl': 2656,
            # 'tresnetl_v2': 2048,
        }
        num_channels = NCDICT[name]
        super().__init__(backbone, train_backbone, num_channels, return_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, args=None):
        super().__init__(backbone, position_embedding)
        # self.args = args
        if args is not None and 'interpotaion' in vars(args) and args.interpotaion:
            self.interpotaion = True
        else:
            self.interpotaion = False


    def forward(self, input: Tensor):
        xs = self[0](input)
        out: List[Tensor] = []
        pos = []
        if isinstance(xs, dict):
        # if torch.jit.isinstance(xs, Dict[str, Tensor]): #TODO
            for name, x in xs.items():
                out.append(x)
                # position encoding
                pos.append(self[1](x).to(x.dtype))
        else:
            # for swin Transformer
            out.append(xs)
            pos.append(self[1](xs).to(xs.dtype))
        return out, pos


def build_backbone(args, logger=None):
    position_embedding = build_position_encoding(args)
    train_backbone = True
    return_interm_layers = False
    # logger.info("build_backbone: sto per chiamare Backbone()")
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, True, args.pretrained, logger=logger)
    bb_num_channels = backbone.num_channels
    model = Joiner(backbone, position_embedding, args)
    model.num_channels = bb_num_channels
    return model
