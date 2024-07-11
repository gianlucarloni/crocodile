'''
This project is the code base for our paper

    "CROCODILE: Causality aids RObustness via COntrastive DIsentangled Learning"
    Carloni, G., Tsaftaris, S. A., & Colantonio, S. (2024).

where we propose a new deep learning framework to tackle domain shift bias on medical image classifiers.

https://github.com/gianlucarloni/crocodile

'''
import argparse

import os, sys
import _init_paths 

# As we'll work with torch DistributedDataParallel (DDP), we need to set the environment variable MPLCONFIGDIR, related to matplotlib, to a specific directory path.
# This is important when, for instance, we use multi-node multi-GPU training with DDP because it helps avoid conflicts when multiple processes are running on different nodes.
os.environ['MPLCONFIGDIR'] = _init_paths.configs_path

import random
import datetime
import time
from typing import List
import json
import numpy as np
from tqdm.auto import tqdm

from sklearn.metrics import roc_auc_score

import torch 
import torch.nn as nn
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
# import torch.multiprocessing as mp # when using single-node multi-gpu, we can use torch.multiprocessing.spawn to spawn multiple processes that run the main_worker function. See the __main__ at the bottom of this script.
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

## We organized all the important scripts for handling dataset preparation, model design, and utility functions into the 'lib' folder.
# Therefore, now we need to access all of those subfolders: lib/dataset, lib/models, and lib/utils:
from dataset.get_dataset import get_datasets
from utils.logger import setup_logger
import models
import models.aslloss
from models.net import build_net
from collections import OrderedDict
from utils.config_ import get_raw_dict

from dataset.cxr_datasets import cate # these are the radiological findings categories, ie, the classes

# torch.autograd.set_detect_anomaly(True) # Set this to true, for debugging purposes only: it checks for expoding/vanishing gradients, or NaN values within Tensors...

mydir=_init_paths.pretrainedmodels_path # where to save/load the pretrained models' weights
torch.hub.set_dir(mydir)
os.environ['TORCH_HOME']=mydir

def parser_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--useCrocodile', action='store_true', default=False, help='Utilize the CROCODILE setting. Default is False. ')
    parser.add_argument('--note', help='note', default='CROCODILE 2024')
    parser.add_argument('--dataname', help='dataname', default='nih')
    parser.add_argument('--dataset_dir', help='dir of dataset', default='./data')
    parser.add_argument('--img_size', default=224, type=int,
                        help='size of input images')
    parser.add_argument('--output', default='myOutput', metavar='DIR', type=str,
                        help='path to output folder')  #TODO #<<<<------------<<<---------------<<---------------<-------customize your folder-------
    parser.add_argument('--num_class', default=9, type=int, # 
                        help="Number of disease-prediction classes, such as the nine radiological findings in the CXR datasets. Technically, it is the number of query slots")
    parser.add_argument('--num_class_crocodile', default=3, type=int,
                        help="number of different source dataset/domain at your disposal")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=True,
                        help='use pre-trained model. default is False. ')
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'],
                        help='which optim to use')
    parser.add_argument('--dropoutrate_randomaddition', default=0.35, type=float,
                        help="Dropout probability to zero out elements in the Confounding feature set to be randomly added to the causal one")
    parser.add_argument('--subset_take1everyN', default=1, type=int)

    #TODO 
    parser.add_argument("--useContrastiveLoss",action='store_true', default=False, help="contrastive loss for enforcing similarity/consistency at image-level and minibatch-level during training")
    parser.add_argument("--contrastiveLossSpace",type=str,default="representation",choices=["representation","activation"])
    parser.add_argument("--crocodile_CODI_common_dim",type=int, default=16, help="Not fully tested yet. It is the commond latent dimension of the mapping space when the contrastiveLossSpace is 'activation' instead of 'representation'")
    parser.add_argument("--useCausalityMaps",action='store_true', default=False)

    parser.add_argument('--w_CE_y', type=float, help='weight for the loss term concerning CrossEntropy on the y (disease) target.',default=0.96)
    parser.add_argument('--w_KL_y',type=float, help='weight for the loss term concerning KL diveergence on the y (disease) target.',default=0.42)
    parser.add_argument('--w_CE_y_bd', type=float,help='weight for the loss term concerning back-door intervened (causality) CrossEntropy on the y (disease) target.',default=0.93)
    parser.add_argument('--w_CE_d',type=float, help='weight for the loss term concerning CrossEntropy on the d (domain/dataset) target.',default=0.96)
    parser.add_argument('--w_KL_d',type=float, help='weight for the loss term concerning KL diveergence on the d (domain/dataset) target.',default=0.42)
    parser.add_argument('--w_CE_d_bd', type=float,help='weight for the loss term concerning back-door intervened (causality) CrossEntropy on the d (domain/dataset)  target.',default=0.93)
    parser.add_argument('--w_Y_imgLvl',type=float, help='weight for the image-level contrastive loss between causal(spurious) disease features and spurious(causal) domain features',default=0.68)
    parser.add_argument('--w_Yca_sameY', type=float,help='weight for the minibatch-level contrastive loss among causal disease features for same-disease samples',default=0.32)
    parser.add_argument('--w_Ysp_diffY', type=float,help='weight for the minibatch-level contrastive loss among spurious disease features for different-disease samples',default=0.32)
    parser.add_argument('--w_Dca_sameD',type=float, help='weight for the minibatch-level contrastive loss among causal domain features for same-domain samples',default=0.32)
    parser.add_argument('--w_Dsp_diffD',type=float, help='weight for the minibatch-level contrastive loss among spurious domain features for different-domain samples',default=0.32)
    parser.add_argument('--w_Yprior',type=float, help='weight for the MSE loss term concerning prior knowledge about the disease targets, y, in the form of causality maps (causal graph, cause-effect relationships).',default=0.5)

    parser.add_argument('--eps', default=1e-5, type=float,
                        help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--dtgfl', action='store_true', default=False,
                        help='disable_torch_grad_focal_loss in asl')
    parser.add_argument('--gamma_pos', default=0, type=float,
                        metavar='gamma_pos', help='gamma pos for simplified asl loss')
    parser.add_argument('--gamma_neg', default=2, type=float,
                        metavar='gamma_neg', help='gamma neg for simplified asl loss')
    parser.add_argument('--loss_dev', default=-1, type=float,
                        help='scale factor for loss')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--val_interval', default=1, type=int, metavar='N',
                        help='interval of validation')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs')
    parser.add_argument('--lr', '--learning-rate', default=0.000004, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=False,
                        help='evaluate model on validation set')
    parser.add_argument('--tSNE', dest='tSNE', action='store_true', default=False,
                        help='compute the tsne visualization on the inference set at choice')
    parser.add_argument('--tSNE_selfreportedrace',dest='tSNE_selfreportedrace', action='store_true', default=False,
                        help='compute the tsne visualization on the Chexpert dataset only, for which we have demographics data on self reported race')
    parser.add_argument('--adjustContrast', dest='adjustContrast', action='store_true', default=False)  

    parser.add_argument('--world_size', type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    # parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    parser.add_argument('--cutout', action='store_true', default=True,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=-1,
                        help='length of the holes. suggest to use default setting -1.')
    parser.add_argument('--cut_fact', type=float, default=0.5,
                        help='mutual exclusion with length. ')

    parser.add_argument('--orid_norm', action='store_true', default=False,
                        help='using mean [0,0,0] and std [1,1,1] to normalize input images')

    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=8192, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=2048, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--keep_other_self_attn_dec', action='store_true',
                        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_input_proj', action='store_true',
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")

    parser.add_argument('--amp', action='store_true', default=False,
                        help='apply amp')
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help='apply early stop')
    parser.add_argument('--kill-stop', action='store_true', default=False,
                        help='apply early stop')
    args = parser.parse_args()

    return args


def get_args():
    args = parser_args()

    # This will be the output folder where the logs, the model's weights, and the tensorboard files will be saved.
    args.output = os.path.join(os.getcwd(), 'out', args.output, time.strftime("%Y%m%d%H", time.localtime(time.time() + 7200))) # adjust the time zone as needed, such as adding 7200 seconds to the current time.
    return args

# We initialize to zero the best values for mean average precision (mAP) score and mean Area Under the Curve (AUC) score for the disease prediction branch obtained with the causally-intervened (back-door) features.
# In particular, the best_meanAUC_c_cap will be used as a reference to save the best model's weights during the validation phase.
best_mAP_c_cap = 0 
best_meanAUC_c_cap = 0


def main(rank, world_size, args):
    """
    Main function for the Crocodile program. We use it to initialize the distributed process group and set up the logger.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        args: The command-line arguments.

    Returns:
        The result of the main_worker function.

    """
   
    print(f'MAIN | Distributed init, rank: {rank}, (worldsize: {world_size}), is CUDA available:{torch.cuda.is_available()}): {args.dist_url}')

    if rank==0: print("Starting init_process_group. The main worker is rank 0.")
    
    torch.distributed.init_process_group(backend='nccl',
                                        init_method=args.dist_url,
                                        world_size=world_size,
                                        rank=rank)
    if rank==0: print("Init_process_group, done.")

    cudnn.benchmark = True #https://stackoverflow.com/a/58965640

    if rank==0: 
        os.makedirs(args.output, exist_ok=True)
        print("Output folder created.")

    logger = setup_logger(output=args.output, distributed_rank= rank, color=False, name="CROCODILE")   
    logger.info("Command: " + ' '.join(sys.argv))
    if dist.get_rank() == 0:
        path = os.path.join(args.output, "config.json")
        with open(path, 'w') as f:
            json.dump(get_raw_dict(args), f, indent=2)
        logger.info("Full configuration saved to {}".format(path))
        logger.info('world size: {}'.format(dist.get_world_size()))
        logger.info('dist.get_rank(): {}'.format(dist.get_rank()))
        logger.info('local_rank: {}'.format(rank))

    return main_worker(rank, world_size, args, logger) # return the main_worker function with the rank, world_size, args, and logger as arguments


def main_worker(rank, world_size, args, logger):
    
    local_rank = int(os.environ['SLURM_LOCALID'])

    global best_mAP_c_cap
    global best_meanAUC_c_cap


    # Let us build our CROCODILE model!
    #   one branch would be the diagnosis predictor as before
    #   the other branch would be the dataset/domain predictor to act as a contrastive regularizer
    

    model = build_net(args, logger) # The args contains the parameters for building the desired architecture. The logger is used to log the architecture details while building it, so that the developer can check/assess the design.
           
    # Set the CUDA device based on the local rank
    torch.cuda.set_device(local_rank)
    # Set the device to CUDA
    device = torch.device("cuda")
    # Move the model to the CUDA device
    model = model.to(device)
    # Wrap the model in the DistributedDataParallel (DDP) module, providing the ID of available device and ID of output device both to the current (local) rank
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)
    
    # Let us define the criterion. We make use of an optimized version of the Asymmetric Loss function, which is a variant of the Focal Loss function.
    criterion = models.aslloss.AsymmetricLossOptimized(
        gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
        disable_torch_grad_focal_loss=args.dtgfl,
        eps=args.eps,
    )

    criterion_crocodile=None # initialize the criterion_crocodile to None, in case we do not use the CROCODILE setting

    if args.useCrocodile: # based on the args, we decide whether to use the CROCODILE setting or not
        criterion_crocodile=nn.CrossEntropyLoss()
        logger.info(f"Set the criterion_crocodile: {criterion_crocodile}")
        
        if args.useContrastiveLoss:
            if args.contrastiveLossSpace=="representation":
                assert args.batch_size % world_size == 0 # Batch size is not divisible by num of gpus.
                
                ## Image-level criterion (the Relational Scorer in Fig. 3 in the paper) is an MSE loss regressing the relational scores r to the ground truths rGT : matched pairs in the Q embedding space have a similarity of 1, and the mismatched pair have a similarity of 0.
                criterion_imageLevel = models.aslloss.CODI_ImageLevel_Loss_embeddingSpace(
                    batch_size= int(args.batch_size / world_size),
                    num_class=args.num_class,
                    num_class_crocodile=args.num_class_crocodile,
                    hidden_dim=args.hidden_dim
                )

                ## Mini-Batch criterion (Equations 4 and 5 in the paper)
                # the terms to enforce consistency/similarity at the mini-batch level of: same-disease samples, same-domain samples
                criterion_same_class = models.aslloss.CODI_MiniBatch_Loss(negative_or_positive_pair='positive', contrastiveLossSpace="representation")
                # the terms to enforce consistency/similarity at the mini-batch level of: different-disease samples, different-domain samples
                criterion_different_class = models.aslloss.CODI_MiniBatch_Loss(negative_or_positive_pair='negative', contrastiveLossSpace="representation")

                # If we decide to use the full power of CROCODILE, we can also add a Task-Prior criterion to the Loss function to inject prior medical knowledge about the task
                if args.useCausalityMaps:
                    criterion_task_prior = models.aslloss.Task_Prior_Loss(categories=cate, batch_size=int(args.batch_size/world_size), device=device)
                    logger.info(f"The Loss function also contains a Task-Prior Loss:\t{criterion_task_prior}")
            
            # elif args.contrastiveLossSpace=="activation":
            #     # For future work or comparison, we allow for an alternative way to use the contrastive loss, this time not in the embedding space (Q features) but instead in the activation space (logits)
            #     # Anyway, in the paper we report the results using the embedding space, so this is just an alternative way to use the contrastive loss.
            #     # Please, consider that this option is not fully tested and may not work as expected. Please, use the embedding space option above.
            #     ## Image-level Loss term
            #     criterion_imageLevel = models.aslloss.CODI_ImageLevel_Loss(
            #         in_features_A = args.num_class, #these are the radiological findings labels, e.g., 9 labels
            #         in_features_B = args.num_class_crocodile, #these are the domain/dataset labels, e.g., 3 labels
            #         common_dim = args.crocodile_CODI_common_dim
            #     )
            #     ## Mini-Batch Loss terms
            #     criterion_same_class = models.aslloss.CODI_MiniBatch_Loss(negative_or_positive_pair='positive', contrastiveLossSpace="activation")
            #     criterion_different_class = models.aslloss.CODI_MiniBatch_Loss(negative_or_positive_pair='negative', contrastiveLossSpace="activation")
            else: raise NotImplementedError
        else: print("Not using contrastive loss...")
            
    
    else: print("Not using CROCODILE setting...")    

    # Let us build our desired optimizer
    args.lr_mult = 1 # as of now, we do not need any multiplier for the learning rate. Adjust as needed.

    if args.optim == 'AdamW':
        param_dicts = [
            {"params": [p for n, p in model.module.named_parameters() if p.requires_grad]},
        ]
        optimizer = getattr(torch.optim, args.optim)(
            param_dicts,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )
    elif args.optim == 'Adam_twd': # Option without weight-decay
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(
            parameters,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )
    else: raise NotImplementedError

    # Let us create a Tensorboard writer object to track everything we need and visualize them with TensorboardX
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output)
        # For instance, here, we add a string text containing the settings of the experiment
        summary_writer.add_text('Settings',f'WORLD_SIZE:{world_size},\
                                    backbone:{args.backbone},\
                                    is_evaluate:{args.evaluate}, \
                                    is_resume:{args.resume}, \
                                    optim:{args.optim},\
                                    num_class:{args.num_class},\
                                    img_size:{args.img_size},\
                                    batch-size:{args.batch_size},\
                                    subset_take1everyN:{args.subset_take1everyN},\
                                    seed:{args.seed},\
                                    epochs:{args.epochs},\
                                    weight-decay:{args.weight_decay},\
                                    lr:{args.lr},\
                                    dropoutrate_randomaddition:{args.dropoutrate_randomaddition},\
                                    early-stop:{args.early_stop},\
                                    pretrained:{args.pretrained},\
                                    useCrocodile:{args.useCrocodile},\
                                    num_class_crocodile:{args.num_class_crocodile},\
                                    useContrastiveLoss:{args.useContrastiveLoss},\
                                    crocodile_CODI_common_dim:{args.crocodile_CODI_common_dim},\
                                    contrastiveLossSpace:{args.contrastiveLossSpace},\
                                    useCausalityMaps:{args.useCausalityMaps},\
                                    w_CE_y:{args.w_CE_y},\
                                    w_KL_y:{args.w_KL_y},\
                                    w_CE_y_bd:{args.w_CE_y_bd},\
                                    w_CE_d:{args.w_CE_d},\
                                    w_KL_d:{args.w_KL_d},\
                                    w_CE_d_bd:{args.w_CE_d_bd},\
                                    w_Y_imgLvl:{args.w_Y_imgLvl},\
                                    w_Yca_sameY:{args.w_Yca_sameY},\
                                    w_Ysp_diffY:{args.w_Ysp_diffY},\
                                    w_Dca_sameD:{args.w_Dca_sameD},\
                                    w_Dsp_diffD:{args.w_Dsp_diffD},\
                                    w_Yprior:{args.w_Yprior}')
    else:
        summary_writer = None

    # Training deep neural networks with large-scale dataset might bring to experiments inadvertently taking a long time to complete, or even crashing due to memory issues.
    # For this reason, we allow the possibility to optionally resume from a checkpoint (for instance, the model's weights saved at the last validation epoch before it was interrupted).
    if args.resume:
        if os.path.isfile(args.resume): ## "./out/myexperiments/2024060823/best.pth.tar"
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(device=device))

            if 'state_dict' in checkpoint:
                state_dict = clean_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                state_dict = clean_state_dict(checkpoint['model'])
            else:
                logger.info("No model or state_dict Found!!!")
                raise ValueError("No model or state_dict Found!!!")
            logger.info("Omitting {}".format(args.resume_omit))
            for omit_name in args.resume_omit:
                del state_dict[omit_name]
            model.module.load_state_dict(state_dict, strict=False)
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
            del checkpoint
            del state_dict
            torch.cuda.empty_cache()
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    if rank==0: print(f"Getting datasets... with args.useCrocodile={args.useCrocodile}")    
    train_dataset, val_dataset, _ = get_datasets(args) # note that we leave the third dataset as an underscore, as we do not need the external test set in the training phase

    logger.info(f"train_dataset: {len(train_dataset)}")   
    logger.info(f"val_dataset: {len(val_dataset)}")       

    ## For debugging purpose only
    # When working with huge image datasets, it could be useful to take a subset of the whole dataset to arrive at some conclusion faster in order to have a sanity check on the correct execution of the code.
    # Attention: that procedure could yield some problems with the AUROC computing in the case the few samples considered are all of the same class.
    #            In addition, it could raise ValueError of the form "ValueError: Number of classes in y_true not equal to the number of columns in 'y_score'" if, for instance, by taking such subsets you incur in the scenario where on the validation/test subset you're computing the metrics for a class that was not present in the training subset!
    # So, assuming your dataset is well balanced and you performed the splitting accurately, consider using a low value for this argument, such as 2 o 3.
    subset_interval=args.subset_take1everyN # take one every N samples in the dataset
    if subset_interval != 1: # One (1) is the normal condition mode, which takes every image in the dataset
        idx_trainval = list(range(0, len(train_dataset), subset_interval))
        idx_test = list(range(0, len(val_dataset), subset_interval))
        train_dataset = torch.utils.data.Subset(train_dataset, idx_trainval)
        val_dataset = torch.utils.data.Subset(val_dataset, idx_test)
        if dist.get_rank() == 0:
            logger.info('DEBUGGING: using subset of datasets, 1 every {} samples; thus, the len() of the dataset/dataloader is different than shown above'.format(subset_interval))
        ####
    
    # Create the data loaders for training set and validation set
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,shuffle=True, drop_last=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(args.batch_size / world_size), shuffle=False, pin_memory=True, num_workers=args.workers, sampler=train_sampler, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(args.batch_size / world_size), shuffle=False,pin_memory=True, num_workers=args.workers, sampler=val_sampler, drop_last=True)
    print(f"train_loader: {len(train_loader)}")
    print(f"val_loader: {len(val_loader)}")
    
    if args.evaluate:
        # This is the evaluation phase, where we can evaluate the model on the external test set
        _, _, test_dataset = get_datasets(args)
        print(f"test_dataset: {len(test_dataset)}")

        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.workers, sampler=test_sampler, drop_last=True)
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=int(args.batch_size / world_size), shuffle=False, pin_memory=True, num_workers=args.workers, sampler=test_sampler, drop_last=True)
        print(f"test_loader: {len(test_loader)}")

        # Now, we can run the validate procedure on the test set (test_loader), setting to None the criterion objects that relate to the Domain/Dataset prediction branch, since we do not leverage it at inference time.
        _, mAP_x, meanAUC_x, mAP_c_cap, meanAUC_c_cap, mAP_c, meanAUC_c, _, _, _, _, _, _, _, _, _, _, _ = validate(test_loader, model, criterion, args, logger, epoch, summary_writer, criterion_crocodile=None, criterion_imageLevel=None, criterion_same_class=criterion_same_class, criterion_different_class=criterion_different_class, criterion_task_prior=criterion_task_prior) 
        # note that in the above output we do not retrieve some variables that are necessary only during training, indicated with underscore (_), to save time and memory.
        
        logger.info(f"||| EXTERNAL TEST SET\nmAP (x):\t{mAP_x}\nmAP (c_cap):\t{mAP_c_cap}\nmAP (c):\t{mAP_c}\n\nmeanAUC_x:\t{meanAUC_x}\nmeanAUC_c_cap:\t{meanAUC_c_cap}\nmeanAUC_c:\t{meanAUC_c}")      
        return

    # We set some tracking variables by means of AverageMeters, for instace to keep track of the loss, the mAP, the time, etc.
    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    mAPs_x = AverageMeter('mAP_x', ':5.5f', val_only=True)
    mAPs_c_cap = AverageMeter('mAP_c_cap', ':5.5f', val_only=True)
    mAPs_c = AverageMeter('mAP_c', ':5.5f', val_only=True)

    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, losses, mAPs_x, mAPs_c_cap,mAPs_c],
        prefix='=> Epoch: ')

    # Let us define a learning rate (LR) scheduler. Here, we exploit the popular one cycle learning rate. Play with the 'max_lr' and 'pct_start' hyperparameters to find the best setting for your task.
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                        epochs=args.epochs, pct_start=0.3)
    #           Alternatively, you could try using the MultiStepLR scheduler, which is more traditional, but remember to update/step it after the epoch, not the minibatch. You can set the milestones and the gamma hyperparameters as needed.

    end = time.time()    
    best_regular_meanAUC_c_cap = 0
    best_epoch_c_cap = -1
    best_regular_epoch_c_cap = -1
    regular_mAP_c_cap_list = regular_mAP_c_list = regular_mAP_x_list =[]

    torch.cuda.empty_cache()    

    # Let us start the training phase: looping over epochs
    for epoch in range(args.start_epoch, args.epochs):

        # When working with Distributed Data Parallel (DDP), it is important to set the epoch to the sampler because it ensures that each replica of the model uses a different random ordering for each epoch.
        train_sampler.set_epoch(epoch)        
        torch.cuda.empty_cache()              

        # We are training the model for one epoch (call to the train() function), tracking down the start time and end time, and returning the loss value.
        startt = time.time()
        if args.useCrocodile:
            if args.useContrastiveLoss:
                if args.useCausalityMaps: #This is the CROCODILE framework at its full power: we have the disease prediction branch, the domain/dataset branch, we use the contrastive losses and the TaskPrior                 
                    loss = train(train_loader, model, None, criterion, optimizer, scheduler, epoch, args, logger, summary_writer, criterion_crocodile=criterion_crocodile, criterion_imageLevel=criterion_imageLevel, criterion_same_class=criterion_same_class,criterion_different_class=criterion_different_class, criterion_task_prior=criterion_task_prior)
                else: # This is the ablated CROCODILE where we remove the TaskPrior
                    loss = train(train_loader, model, None, criterion, optimizer, scheduler, epoch, args, logger, summary_writer, criterion_crocodile=criterion_crocodile, criterion_imageLevel=criterion_imageLevel, criterion_same_class=criterion_same_class,criterion_different_class=criterion_different_class, criterion_task_prior=None)
            else: # This is the ablated CROCODILE where we remove also the contrastive losses
                loss = train(train_loader, model, None, criterion, optimizer, scheduler, epoch, args, logger, summary_writer, criterion_crocodile=criterion_crocodile, criterion_imageLevel=None, criterion_same_class=None, criterion_different_class=None, criterion_task_prior=None)
        else: # This is the baseline, one-branch model 
            loss = train(train_loader, model, None, criterion, optimizer, scheduler, epoch, args, logger, summary_writer, criterion_crocodile=None, criterion_imageLevel=None, criterion_same_class=None, criterion_different_class=None, criterion_task_prior=None)

        endt = time.time()
        logger.info("Elapsed time (training):    {} hours".format((endt - startt)/3600))

        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % args.val_interval == 0:

            ## evaluate on the (internal) validation set. This is the dataset we use to iteratively validate the model and tune the hyperparameters. Thus, it is not the external never-before-seen test set.
            if args.useCrocodile:
                if args.useContrastiveLoss:
                    if args.contrastiveLossSpace=="representation":
                        if args.useCausalityMaps: # disease branch + domain branch + contrastive losses + TaskPrior
                            loss, mAP_x, meanAUC_x, mAP_c_cap, meanAUC_c_cap, mAP_c, meanAUC_c, Q, Q_bar, Q_crocodile, Q_bar_crocodile, cmap_Q, AP_x, AP_c_cap, AP_c, auc_scores_x, auc_scores_c_cap, auc_scores_c = validate(val_loader, model, criterion, args, logger, epoch, summary_writer, criterion_crocodile=criterion_crocodile, criterion_imageLevel=criterion_imageLevel, criterion_same_class=criterion_same_class, criterion_different_class=criterion_different_class, criterion_task_prior=criterion_task_prior) 
                        else:
                            loss, mAP_x, meanAUC_x, mAP_c_cap, meanAUC_c_cap, mAP_c, meanAUC_c, Q, Q_bar, Q_crocodile, Q_bar_crocodile, _, AP_x, AP_c_cap, AP_c, auc_scores_x, auc_scores_c_cap, auc_scores_c = validate(val_loader, model, criterion, args, logger, epoch, summary_writer, criterion_crocodile=criterion_crocodile, criterion_imageLevel=criterion_imageLevel, criterion_same_class=criterion_same_class, criterion_different_class=criterion_different_class, criterion_task_prior=None) 
                    else: #"activation" (logit) space (not fully tested, not used in the paper)
                        loss, mAP_x, meanAUC_x, mAP_c_cap, meanAUC_c_cap, mAP_c, meanAUC_c, _, _, _, _, _, AP_x, AP_c_cap, AP_c, auc_scores_x, auc_scores_c_cap, auc_scores_c = validate(val_loader, model, criterion, args, logger, epoch, summary_writer, criterion_crocodile=criterion_crocodile, criterion_imageLevel=criterion_imageLevel, criterion_same_class=criterion_same_class, criterion_different_class=criterion_different_class, criterion_task_prior=None) 
                else: #caumed+ domain/dataset branch
                    loss, mAP_x, meanAUC_x, mAP_c_cap, meanAUC_c_cap, mAP_c, meanAUC_c, _, _, _, _, _, AP_x, AP_c_cap, AP_c, auc_scores_x, auc_scores_c_cap, auc_scores_c = validate(val_loader, model, criterion, args, logger, epoch, summary_writer, criterion_crocodile=criterion_crocodile, criterion_imageLevel=None, criterion_same_class=None, criterion_different_class=None, criterion_task_prior=None) 
            else: #baseline, caumed
                loss, mAP_x, meanAUC_x, mAP_c_cap, meanAUC_c_cap, mAP_c, meanAUC_c, _, _, _, _, _, AP_x, AP_c_cap, AP_c, auc_scores_x, auc_scores_c_cap, auc_scores_c = validate(val_loader, model, criterion, args, logger, epoch, summary_writer, criterion_crocodile=None, criterion_imageLevel=None, criterion_same_class=None, criterion_different_class=None, criterion_task_prior=None) 

            # Note on the above code that, depending on the settings, we may not use all the variables returned by the validate() function. Save time and memory wheneve possible.
            
            # Let us update the tracking variables with the obtained values for the loss and average precision scores, but also for the time and estimated time left.
            losses.update(loss)
            mAPs_x.update(mAP_x)
            mAPs_c_cap.update(mAP_c_cap)
            mAPs_c.update(mAP_c)        
            epoch_time.update(time.time() - end)
            end = time.time()
            eta.update(epoch_time.avg * (args.epochs - epoch - 1))
        
            progress.display(epoch, logger)

            if summary_writer:
                # Again, we believe logging everything to the tensorboard logger is a good practice. Let's add the values for the loss and for the mean average precision and AUC scores corresponding to each of the three classifiers: x (the causal), c_cap (the backdoor internvened), c (the confounding/spurious).
                summary_writer.add_scalar('val_loss', loss, epoch)

                summary_writer.add_scalar('val_mAP_x', mAP_x, epoch)
                summary_writer.add_scalar('val_mAP_c_cap', mAP_c_cap, epoch)
                summary_writer.add_scalar('val_mAP_c', mAP_c, epoch)                
                
                summary_writer.add_scalar('val_meanAUC_x', meanAUC_x, epoch)
                summary_writer.add_scalar('val_meanAUC_c_cap', meanAUC_c_cap, epoch)
                summary_writer.add_scalar('val_meanAUC_c', meanAUC_c, epoch)


            ## We base the following saving of the model on the validation AUC score obtained with the backdoor intervened classifier at this epoch
            if meanAUC_c_cap > best_regular_meanAUC_c_cap:
                best_regular_meanAUC_c_cap = meanAUC_c_cap # this is the best regular, non-EMA
                best_regular_epoch_c_cap = epoch
 
            state_dict = model.state_dict()    # we get the state dictionary of the model anyway.
                       
            # Is this model the best model globally? Then
            is_best = meanAUC_c_cap > best_meanAUC_c_cap
            if is_best:
                best_epoch_c_cap = epoch
            best_meanAUC_c_cap = max(meanAUC_c_cap, best_meanAUC_c_cap)
            logger.info("{} | Set best meanAUC (c_cap) {} in ep {}".format(epoch, best_meanAUC_c_cap, best_epoch_c_cap))
            logger.info("  | best regular (non-EMA) meanAUC (c_cap) {} in ep {}".format(best_regular_meanAUC_c_cap, best_regular_epoch_c_cap))
            # Save the model checkpoint only if the process rank is 0 (the master process)
            if dist.get_rank() == 0:
                save_checkpoint(
                    {
                    'epoch': epoch + 1,
                    'state_dict': state_dict,
                    'best_meanAUC_c_cap': best_meanAUC_c_cap,
                    'optimizer': optimizer.state_dict(),
                    }, is_best=is_best, filename=os.path.join(args.output, 'c_cap.pth.tar')
                )

                # in case it is the best model, we also save the numpy objects for the AP and AUC scores, and the pt objects for the Q features, for offline processing/analysis/visualization if needed.
                if is_best:
                    np.save(os.path.join(args.output, "AP_x_best.npy"), AP_x) # beware that this naming overrides the filenames whenever it is a better model:
                    np.save(os.path.join(args.output, "AP_c_cap_best.npy"), AP_c_cap)
                    np.save(os.path.join(args.output, "AP_c_best.npy"), AP_c)

                    np.save(os.path.join(args.output, "auc_scores_x.npy"), auc_scores_x)
                    np.save(os.path.join(args.output, "auc_scores_c_cap.npy"), auc_scores_c_cap)
                    np.save(os.path.join(args.output, "auc_scores_c.npy"), auc_scores_c)
                    logger.info("Saved numpy objects for AP and AUC scores.")
                    if args.useCrocodile and args.useContrastiveLoss and args.contrastiveLossSpace=="representation":
                        torch.save(Q, os.path.join(args.output, f"Q{epoch}.pt"))  # We preserve the epoch value, eventually useful to visualize the evolution of Q features over time
                        torch.save(Q_bar, os.path.join(args.output, f"Q_bar{epoch}.pt")) 
                        torch.save(Q_crocodile, os.path.join(args.output, f"Q_crocodile{epoch}.pt")) 
                        torch.save(Q_bar_crocodile, os.path.join(args.output, f"Q_bar_crocodile{epoch}.pt")) 
                        logger.info("Saved pt objects for Q, Qbar, Qcrocodile, Qbarcrocodile.")
                        if args.useCausalityMaps:
                            torch.save(cmap_Q, os.path.join(args.output, f"cmap_Q{epoch}.pt"))
                            logger.info("Saved pt object for cmap_Q.")

            # In case you specified the early_stop option, we can stop the training if the best epoch is too far from the current epoch
            if args.early_stop:
                if best_epoch_c_cap >= 0 and epoch - max(best_epoch_c_cap, best_regular_epoch_c_cap) >= 8:
                    logger.info("Difference between epoch - best_epoch = {}, stop!".format(epoch - best_epoch_c_cap))
                    if dist.get_rank() == 0 and args.kill_stop:
                        filename = sys.argv[0].split(' ')[0].strip()
                        killedlist = kill_process(filename, os.getpid())
                        logger.info("Kill all process of {}: ".format(filename) + " ".join(killedlist))
                    break

    # Cool, we've reached the end!
    print("---End of training---")
    # Just before closing the session, we need to clean the process group in DDP, and flush/close the writer to make sure that all pending events have been written to disk.
    dist.destroy_process_group()   
    if summary_writer:
        summary_writer.flush() 
        summary_writer.close()
    
    # When working with multi-node multi-GPU training, it is important to ensure that all processes are terminated correctly. Depending on the environment, you may need to kill the processes manually.
    # Therefore, here, we use an undefined function to force the exit of the processes in case they are not correctly manged before, eventually leading to the halt of the python script due to some error. That is fine.
    if rank==0:
        force_exit_here
    elif rank>0:
        force_exit_here

    return 0


def train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger, summary_writer=None, criterion_crocodile=None, criterion_imageLevel=None, criterion_same_class=None, criterion_different_class=None,criterion_task_prior=None):

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp) # The GradScaler is used for automatic mixed precision (amp) training, which allows for faster and more memory-efficient computations on supported hardware.

    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        len(train_loader),
        [lr, losses, mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # First, we need to switch our model to training mode
    model.train()

    end = time.time()

    # If you are concerned about memory requirements, availability of graphic memory (GPU) and allocated resources, it might be beneficial to
    # returns the maximum GPU memory occupied by tensors in bytes for this given GPU device. This is done by calling the torch.cuda.max_memory_allocated() function.
    logger.info(f"TRAIN (just before starting epochs) --> Mem {torch.cuda.max_memory_allocated() / (1024.0 * 1024.0* 1024.0)} GB")
    
    for i, (images, target, target_crocodile) in enumerate(tqdm(train_loader)): # we use tqdm to have a progress bar in the terminal for bored users :)

        # We need to move the images and the target to the GPU device
        images = images.cuda(non_blocking=True) # the non_blocking=True flag allows for asynchronous data transfer, which can improve performance when using DDP
        target = target.cuda(non_blocking=True)
        target_crocodile=target_crocodile.cuda(non_blocking=True)

        # let us define some variables to keep track of the loss values. Initialize them to zero.
        loss_caumed=0
        loss1=0
        loss2=0
        loss3=0
        loss_crocodile=0
        loss1_crocodile=0
        loss2_crocodile=0
        loss3_crocodile=0
        loss_imageLevel=0
        loss_Y_ca_D_sp=0
        loss_Y_sp_D_ca=0
        loss_minibatchLevel=0
        loss_Y_ca_sameY=0
        loss_Y_sp_differentY=0
        loss_D_ca_sameD=0
        loss_D_sp_differentD=0
        loss_task_prior=0

        # Now, we compute the output of the model on the input images. We use the automatic mixed precision (amp) context manager to speed up the computations.
        with torch.cuda.amp.autocast(enabled=args.amp):          
            # If we are using the CROCODILE framework, we need to consider the contrastive losses and the TaskPrior, if they are enabled.
            if args.useCrocodile:
                if args.useContrastiveLoss:                       
                    if args.contrastiveLossSpace == "activation": # This logit level (activation space) is not fully tested, not used in the paper. Please, use the "representation" option. 
                        z_x, z_c_cap, z_c, z_x_crocodile, z_c_cap_crocodile, z_c_crocodile, _, _, _, _, _ = model(images)

                        loss_Y_ca_D_sp = criterion_imageLevel(z_x, z_c_crocodile) #causal features for diagnosis, spurious features for domain/dataset
                        loss_Y_sp_D_ca = criterion_imageLevel(z_c, z_x_crocodile) #spurious features for diagnosis, causal features for domain/dataset
                        loss_Y_ca_sameY = criterion_same_class(z_x, target) #enforce alignment between the causal features for diagnosis among same-disease samples #TODO implement prior kno: perfect and almost-perf matches
                        loss_Y_sp_differentY = criterion_different_class(z_c, target) #spurious features for diagnosis among different-disease samples
                        loss_D_ca_sameD = criterion_same_class(z_x_crocodile, target_crocodile) #causal features for dataset among same-dataset samples
                        loss_D_sp_differentD = criterion_different_class(z_c_crocodile, target_crocodile) # spurious features for dataset among different-dataset samples
                    
                    elif args.contrastiveLossSpace == "representation":  
                        if args.useCausalityMaps:
                            # This is the case where we use the causality maps to inject prior knowledge about the task. Indeed, we return the cmap_Q object from the model.
                            z_x, z_c_cap, z_c, z_x_crocodile, z_c_cap_crocodile, z_c_crocodile, Q, Q_bar, Q_crocodile, Q_bar_crocodile, cmap_Q = model(images)  

                            # Inject prior knowledge by means of the causality maps, utilizing the TaskPrior loss term                              
                            loss_task_prior = criterion_task_prior(cmap_Q)
                            loss_task_prior = args.w_Yprior*loss_task_prior # modulate the loss by a weight factor
                        else:
                            # in this case, we disregard the causality maps and the TaskPrior loss term
                            z_x, z_c_cap, z_c, z_x_crocodile, z_c_cap_crocodile, z_c_crocodile, Q, Q_bar, Q_crocodile, Q_bar_crocodile, _ = model(images) 
                        
                        # In the following, consider that the objects Q_ are of shape torch.Size([batch_size, num_class, hidden_dim]), so, for instance,
                        #                   Q: [12, 9, 2048], Q_bar: [12, 9, 2048], Q_crocodile: [12, 3, 2048], Q_bar_crocodile: [12, 3, 2048]

                        ## Image-level contrastive loss ('Learning to Attend'): this is our proposed Relational Scorer (RS, Fig. 3.) module, which takes as input the concatenation of the two features to be compared, and outputs a scalar score.
                        loss_imageLevel = args.w_Y_imgLvl*criterion_imageLevel(Q, Q_bar, Q_crocodile, Q_bar_crocodile)    # this is as if we did 'loss_Y_ca_D_sp' and 'loss_Y_sp_D_ca' in a single pass 

                        ## Minibatch-level contrastive losses:            
                        loss_Y_ca_sameY = criterion_same_class(Q, target) #enforce alignment between the causal features for diagnosis among same-disease samples
                        loss_Y_sp_differentY = criterion_different_class(Q_bar, target) #spurious features for diagnosis among different-disease samples
                        loss_D_ca_sameD = criterion_same_class(Q_crocodile, target_crocodile) #causal features for dataset among same-dataset samples
                        loss_D_sp_differentD = criterion_different_class(Q_bar_crocodile, target_crocodile) # spurious features for dataset among different-dataset samples
                    
                    else:
                        logger.info(f"args.contrastiveLossSpace is neither 'activation' nor 'representation', raise ValueError")
                        raise ValueError   
                else:
                    # Ablated version that uses crocodile but not contrastive losses                        
                    z_x, z_c_cap, z_c, z_x_crocodile, z_c_cap_crocodile, z_c_crocodile, _, _, _, _, _ = model(images)
            else:
                # Version that does not use crocodile, thus it is the baseline, regular training with only disease prediction branch.
                z_x, z_c_cap, z_c, _, _, _, _, _, _, _, _ = model(images)


            ####### Disease-prediction branch (Y) #######
            ## The supervised loss regards causal features
            loss1 = criterion(z_x, target) #the supervised loss
            if torch.isnan(loss1).any():
                logger.info("TRAIN Loss - loss1 has NaNs: raise ValueError and exit")
                raise ValueError
            ## The confounding features get associated with a uniform probability over all classes
            z_c_log_sm = F.log_softmax(z_c, dim=1)
            uniform_target = torch.ones_like(z_c_log_sm, dtype=torch.float, device='cuda') / args.num_class #
            loss2 = F.kl_div(z_c_log_sm, uniform_target, reduction='batchmean')
            if torch.isnan(loss2).any():
                logger.info("TRAIN Loss - loss2 has NaNs: raise ValueError and exit")
                raise ValueError            
            ## The causally-intervened features are forced to align with the actual target to make them robust to confounding features
            loss3 = criterion(z_c_cap, target)
            if torch.isnan(loss3).any():
                logger.info("TRAIN Loss - loss3 has NaNs: raise ValueError and exit")
                raise ValueError            
            # Embedding all the losses into a single loss term, we obtain the L_y loss, also named CAUMED (causality in medical imaging), which is Equation (1) in the paper.
            loss_caumed = args.w_CE_y*loss1 + args.w_KL_y*loss2 + args.w_CE_y_bd*loss3
            #######

            ####### Domain-prediction branch (D) #######
            if args.useCrocodile:
                ## The supervised loss regards causal features
                loss1_crocodile = criterion_crocodile(z_x_crocodile, target_crocodile) #the supervised loss
                if torch.isnan(loss1_crocodile).any():
                    logger.info("TRAIN Loss - loss1_crocodile has NaNs: raise ValueError and exit")
                    raise ValueError              
                ## The confounding features get associated with a uniform probability over all classes
                z_c_log_sm_crocodile = F.log_softmax(z_c_crocodile, dim=1)
                uniform_target_crocodile = torch.ones_like(z_c_log_sm_crocodile, dtype=torch.float, device='cuda') / args.num_class_crocodile #
                loss2_crocodile = F.kl_div(z_c_log_sm_crocodile, uniform_target_crocodile, reduction='batchmean')
                if torch.isnan(loss2_crocodile).any():
                    logger.info("TRAIN Loss - loss2_crocodile has NaNs: raise ValueError and exit")
                    raise ValueError            
                ## The causally-intervened features are forced to align with the actual target to make them robust to confounding features
                loss3_crocodile = criterion_crocodile(z_c_cap_crocodile, target_crocodile)
                if torch.isnan(loss3_crocodile).any():
                    logger.info("TRAIN Loss - loss3_crocodile has NaNs: raise ValueError and exit")
                    raise ValueError                           
                
                # Embedding those three losses into a single loss term, we obtain the L_d loss, also named crocodile, which is Equation (2) in the paper.
                loss_crocodile = args.w_CE_d*loss1_crocodile + args.w_KL_d*loss2_crocodile + args.w_CE_d_bd*loss3_crocodile
                
                # Moreover, we can compute the contrastive losses, if they are enabled.
                if args.useContrastiveLoss:
                    # we compute equations (4) and (5) in a single pass, thus we obtain the contrastive loss at minibatch level:
                    loss_minibatchLevel = args.w_Yca_sameY*loss_Y_ca_sameY + args.w_Ysp_diffY*loss_Y_sp_differentY + args.w_Dca_sameD*loss_D_ca_sameD + args.w_Dsp_diffD*loss_D_sp_differentD

                    if args.contrastiveLossSpace=="activation": # this option is not fully tested, not used in the paper. Please, use the "representation" option.
                        #Differently to the 'representation' case, which already have a 'loss_imageLevel', here, we create this new variable by unifying the two terms and give it the same name
                        loss_imageLevel = args.w_Y_imgLvl*loss_Y_ca_D_sp + args.w_Y_imgLvl*loss_Y_sp_D_ca 

                            

            ### Now, we compose the loss terms into the final loss:
            # -----loss_caumed         = args.w_CE_y*loss1 + args.w_KL_y*loss2 + args.w_CE_y_bd*loss3                               # For the disease prediction branch
            # -----loss_crocodile      = args.w_CE_d*loss1_crocodile + args.w_KL_d*loss2_crocodile + args.w_CE_d_bd*loss3_crocodile # For the domain/dataset prediction branch
            # -----loss_imageLevel     = args.w_Y_imgLvl*criterion_imageLevel(Q, Q_bar, Q_crocodile, Q_bar_crocodile)               # For the contrastive loss at image sample level, with the Option "representation"
            # -----loss_minibatchLevel = args.w_Yca_sameY*loss_Y_ca_sameY + args.w_Ysp_diffY*loss_Y_sp_differentY + args.w_Dca_sameD*loss_D_ca_sameD + args.w_Dsp_diffD*loss_D_sp_differentD # For the contrastive loss at minibatch level
            # -----loss_task_prior     = args.w_Yprior*criterion_task_prior(cmap_Q)                                                 # For injecting prior knowledge by means of the causality maps

            ## Let us compose the final training loss.
            # Depending on the settings of the experiment, some of those terms might not be set (value remains 0):
            loss = loss_caumed + loss_crocodile + loss_imageLevel + loss_minibatchLevel + loss_task_prior

            if dist.get_rank() == 0:
                if summary_writer:
                    # tensorboard logger
                    summary_writer.add_scalar('TrainLoss', loss, epoch)
                    summary_writer.add_scalar('TrainLoss/CauMed', loss_caumed, i)
                    summary_writer.add_scalar('TrainLoss/CauMed/CE_y', args.w_CE_y*loss1, i)
                    summary_writer.add_scalar('TrainLoss/CauMed/KL_y', args.w_KL_y*loss2, i)
                    summary_writer.add_scalar('TrainLoss/CauMed/CE_bd_y', args.w_CE_y_bd*loss3, i)
                    summary_writer.add_scalar('TrainLoss/Domain', loss_crocodile, i)
                    summary_writer.add_scalar('TrainLoss/Domain/CE_d', args.w_CE_d*loss1_crocodile, i)
                    summary_writer.add_scalar('TrainLoss/Domain/KL_d', args.w_KL_d*loss2_crocodile, i)
                    summary_writer.add_scalar('TrainLoss/Domain/CE_bd_d', args.w_CE_d_bd*loss3_crocodile, i)
                    summary_writer.add_scalar('TrainLoss/ContrImageLevel', args.w_Y_imgLvl*loss_imageLevel, i)
                    summary_writer.add_scalar('TrainLoss/ContrBatchLevel', loss_minibatchLevel, i)
                    summary_writer.add_scalar('TrainLoss/ContrBatchLevel/Y_ca_sameY', args.w_Yca_sameY*loss_Y_ca_sameY, i)
                    summary_writer.add_scalar('TrainLoss/ContrBatchLevel/Y_sp_differentY', args.w_Ysp_diffY*loss_Y_sp_differentY, i)
                    summary_writer.add_scalar('TrainLoss/ContrBatchLevel/D_ca_sameD', args.w_Dca_sameD*loss_D_ca_sameD, i)
                    summary_writer.add_scalar('TrainLoss/ContrBatchLevel/D_sp_differentD', args.w_Dsp_diffD*loss_D_sp_differentD, i)
                    summary_writer.add_scalar('TrainLoss/TaskPrior', args.w_Yprior*loss_task_prior, i)
                
                if random.random()<0.01: # In addition to writing on the Tensorboard logger, print the loss values every so often (say, 1% of the time)
                    logger.info(f"|\tTRAIN LOSS: {loss}\n\
                                \tCAU-MED:\t{loss_caumed}\n\
                                \t\targs.w_CE_y*loss1:\t{args.w_CE_y}*{loss1}=\t{args.w_CE_y*loss1}\n\
                                \t\targs.w_KL_y*loss2:\t{args.w_KL_y}*{loss2}=\t{args.w_KL_y*loss2}\n\
                                \t\targs.w_CE_y_bd*loss3:\t{args.w_CE_y_bd}*{loss3}=\t{args.w_CE_y_bd*loss3}\n\
                                \tDOMAIN/DATASET:\t{loss_crocodile}\n\
                                \t\targs.w_CE_d*loss1_crocodile:\t{args.w_CE_d}*{loss1_crocodile}=\t{args.w_CE_d*loss1_crocodile}\n\
                                \t\targs.w_KL_d*loss2_crocodile:\t{args.w_KL_d}*{loss2_crocodile}=\t{args.w_KL_d*loss2_crocodile}\n\
                                \t\targs.w_CE_d_bd*loss3_crocodile:\t{args.w_CE_d_bd}*{loss3_crocodile}=\t{args.w_CE_d_bd*loss3_crocodile}\n\
                                \tCONTR-ImageLevel:\t{loss_imageLevel}\n\
                                \t\targs.w_Y_imgLvl:\t{args.w_Y_imgLvl}\n\
                                \tCONTR-MinibatchLevel:\t{loss_minibatchLevel}\n\
                                \t\targs.w_Yca_sameY*loss_Y_ca_sameY:\t{args.w_Yca_sameY}*{loss_Y_ca_sameY}=\t{args.w_Yca_sameY*loss_Y_ca_sameY}\n\
                                \t\targs.w_Ysp_diffY*loss_Y_sp_differentY:\t{args.w_Ysp_diffY}*{loss_Y_sp_differentY}=\t{args.w_Ysp_diffY*loss_Y_sp_differentY}\n\
                                \t\targs.w_Dca_sameD*loss_D_ca_sameD:\t{args.w_Dca_sameD}*{loss_D_ca_sameD}=\t{args.w_Dca_sameD*loss_D_ca_sameD}\n\
                                \t\targs.w_Dsp_diffD*loss_D_sp_differentD:\t{args.w_Dsp_diffD}*{loss_D_sp_differentD}=\t{args.w_Dsp_diffD*loss_D_sp_differentD}\n\
                                \tTASK-Prior:\t{loss_task_prior}\n\
                                \t\targs.w_Yprior:\t{args.w_Yprior}")                            
                    
            if args.loss_dev > 0:
                loss = loss*args.loss_dev

        # record loss
        losses.update(loss, images.size(0))
        # update memory usage
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) # in megabytes

        # compute gradient and do the step
        optimizer.zero_grad(set_to_none=True) # This flag instructs the optimizer to set the gradients of all the parameters to None after performing the gradient update. This can save memory, especially when using AMP. Indeed, the default would be storing zero (0), and not None, which can be more memory-consuming.
        scaler.scale(loss).backward() 

        scaler.step(optimizer)
        scaler.update()
        
        # delete variables to free memory and empy the cache
        del images, target, loss, loss1, loss2, loss3 #TODO
        if args.useCrocodile:
            del target_crocodile, loss1_crocodile, loss2_crocodile, loss3_crocodile 
        torch.cuda.empty_cache()
        
        # One cycle learning rate. Note that the scheduler is updated at each iteration (minibatches), not at each epoch like many other schedulers.
        scheduler.step()
        
        lr.update(get_learning_rate(optimizer))

        if i % args.print_freq == 0:
            progress.display(i, logger)
    
    return losses.avg

# Define the validation routine
@torch.no_grad()
def validate(data_loader, model, criterion, args, logger, epoch, summary_writer, criterion_crocodile=None, criterion_imageLevel=None, criterion_same_class=None, criterion_different_class=None, criterion_task_prior=None): #TODO 10 may 2024

    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, losses, mem],
        prefix='Test: ')

    # switch to evaluate mode    
    model.eval()

    saved_data_x = []
    saved_data_c_cap = []
    saved_data_c = []

    targets = None
    targets_crocodile = None

    outputs_x = None
    outputs_c_cap = None
    outputs_c = None

    outputs_x_crocodile = None
    outputs_c_cap_crocodile = None
    outputs_c_crocodile = None
    
    Q = Q_bar = Q_crocodile = Q_bar_crocodile = cmap_Q = None

    with torch.no_grad():
        end = time.time()
        if args.evaluate: # This is the case of an evaluation mode on the external test set for OOD performance (domain generalization, DG). Skip below for the Validation-during-training procedure
            
            for i, (images, target, _) in enumerate(data_loader):     
                # move to GPU           
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)                

                # compute output
                with torch.cuda.amp.autocast(enabled=args.amp):         
                    z_x, z_c_cap, z_c, _, _, _, Q, Q_bar, _, _, cmap_Q = model(images)                                 
                    # Apply the sigmoid function to the logits obtained from the model to obtain the probabilities corresponding to each classifier: x (the causal), c_cap (the backdoor internvened), c (the confounding/spurious).         
                    output_sm_x = torch.sigmoid(z_x) # the causal classifier
                    output_sm_c_cap = torch.sigmoid(z_c_cap) # the backdoor intervened classifier
                    output_sm_c = torch.sigmoid(z_c) # the confounding/spurious classifier 
                
                tar = target.cpu()            
                out_x = output_sm_x.cpu()
                out_c_cap = output_sm_c_cap.cpu()
                out_c = output_sm_c.cpu()
                targets = tar if targets == None else torch.cat([targets, tar])
                outputs_x = out_x if outputs_x == None else torch.cat([outputs_x, out_x])
                outputs_c_cap = out_c_cap if outputs_c_cap == None else torch.cat([outputs_c_cap, out_c_cap])
                outputs_c = out_c if outputs_c == None else torch.cat([outputs_c, out_c])                     

                # let us save some of them to disk
                _item_x = torch.cat((out_x, tar), 1)
                _item_c_cap = torch.cat((out_c_cap, tar), 1)
                _item_c = torch.cat((out_c, tar), 1)
                saved_data_x.append(_item_x)
                saved_data_c_cap.append(_item_c_cap)
                saved_data_c.append(_item_c)              

                mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) # in megabytes

                # measure elapsed time (validation)
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0 and dist.get_rank() == 0:
                    progress.display(i, logger)
        
        else: # VALIDATION during training       
            
            for i, (images, target, target_crocodile) in enumerate(data_loader): #TODO
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                target_crocodile=target_crocodile.cuda(non_blocking=True)          

                loss_caumed=0
                loss1=0
                loss2=0
                loss3=0
                loss_crocodile=0
                loss1_crocodile=0
                loss2_crocodile=0
                loss3_crocodile=0
                loss_imageLevel=0
                loss_Y_ca_D_sp=0
                loss_Y_sp_D_ca=0
                loss_minibatchLevel=0
                loss_Y_ca_sameY=0
                loss_Y_sp_differentY=0
                loss_D_ca_sameD=0
                loss_D_sp_differentD=0
                loss_task_prior=0

                # compute output
                with torch.cuda.amp.autocast(enabled=args.amp):         
                    if args.useCrocodile:
                        if args.useContrastiveLoss:
                            if args.contrastiveLossSpace == "activation": # this option at logit level (activation space) is not fully tested yet, not used in the paper. Please, use the "representation" option.
                                z_x, z_c_cap, z_c, z_x_crocodile, z_c_cap_crocodile, z_c_crocodile, _, _, _, _, _ = model(images)

                                loss_Y_ca_D_sp = criterion_imageLevel(z_x, z_c_crocodile) #causal features for diagnosis, spurious features for domain/dataset
                                loss_Y_sp_D_ca = criterion_imageLevel(z_c, z_x_crocodile) #spurious features for diagnosis, causal features for domain/dataset
                                loss_Y_ca_sameY = criterion_same_class(z_x, target) #enforce alignment between the causal features for diagnosis among same-disease samples
                                loss_Y_sp_differentY = criterion_different_class(z_c, target) #spurious features for diagnosis among different-disease samples
                                loss_D_ca_sameD = criterion_same_class(z_x_crocodile, target_crocodile) #causal features for dataset among same-dataset samples
                                loss_D_sp_differentD = criterion_different_class(z_c_crocodile, target_crocodile) # spurious features for dataset among different-dataset samples
                            
                            elif args.contrastiveLossSpace == "representation":                                  
                                if args.useCausalityMaps:
                                    z_x, z_c_cap, z_c, z_x_crocodile, z_c_cap_crocodile, z_c_crocodile, Q, Q_bar, Q_crocodile, Q_bar_crocodile, cmap_Q = model(images)
                                    loss_task_prior = criterion_task_prior(cmap_Q)
                                    loss_task_prior = args.w_Yprior*loss_task_prior
                                else:    
                                    z_x, z_c_cap, z_c, z_x_crocodile, z_c_cap_crocodile, z_c_crocodile, Q, Q_bar, Q_crocodile, Q_bar_crocodile, _ = model(images)                                 
                                

                                # In the following, consider that the objects Q_ are of shape torch.Size([batch_size, num_class, hidden_dim]), so, for instance,
                                #                Q: [12, 9, 2048], Q_bar: [12, 9, 2048], Q_crocodile: [12, 3, 2048], Q_bar_crocodile: [12, 3, 2048]

                                ## Image-level contrastive loss ('Learning to Attend'): this is our proposed Relational Scorer (RS, Fig. 3.) module, which takes as input the concatenation of the two features to be compared, and outputs a scalar score.
                                loss_imageLevel = criterion_imageLevel(Q, Q_bar, Q_crocodile, Q_bar_crocodile)    # this is as if we did 'loss_Y_ca_D_sp' and 'loss_Y_sp_D_ca' in a single pass             
                                
                                # Minibatch-level contrastive losses:
                                loss_Y_ca_sameY = criterion_same_class(Q, target) #enforce alignment between the causal features for diagnosis among same-disease samples
                                loss_Y_sp_differentY = criterion_different_class(Q_bar, target) #spurious features for diagnosis among different-disease samples
                                loss_D_ca_sameD = criterion_same_class(Q_crocodile, target_crocodile) #causal features for dataset among same-dataset samples
                                loss_D_sp_differentD = criterion_different_class(Q_bar_crocodile, target_crocodile) # spurious features for dataset among different-dataset samples
                            
                            else:
                                logger.info(f"args.contrastiveLossSpace is neither 'activation' nor 'representation', raise ValueError")
                                raise ValueError   
                        else: #use crocodile but not contrastive losses                        
                            z_x, z_c_cap, z_c, z_x_crocodile, z_c_cap_crocodile, z_c_crocodile, _, _, _, _, _ = model(images)
                    else: # do not use crocodile, regular training
                        z_x, z_c_cap, z_c, _, _, _, _, _, _, _, _ = model(images)
        
                    ##### CAU MED block (disease prediction)######                
                    loss1 = criterion(z_x, target) #the supervised loss
                    if torch.isnan(loss1).any():                    
                        logger.info("VALID Loss - loss1 has NaNs: raise ValueError and exit")
                        raise ValueError                
                    uniform_target = torch.ones_like(z_c, dtype=torch.float, device='cuda') / args.num_class #we need to push its prediction equally to all categories
                    loss2 = F.kl_div(F.log_softmax(z_c, dim=1), uniform_target, reduction='batchmean')
                    if torch.isnan(loss2).any():
                        logger.info("VALID Loss - loss2 has NaNs: raise ValueError and exit")
                        raise ValueError                
                    loss3 = criterion(z_c_cap, target)
                    if torch.isnan(loss3).any():
                        logger.info("VALID Loss - loss3 has NaNs: raise ValueError and exit")
                        raise ValueError  
                    
                    loss_caumed = args.w_CE_y*loss1 + args.w_KL_y*loss2 + args.w_CE_y_bd*loss3
                    
                    output_sm_x = torch.sigmoid(z_x)
                    output_sm_c_cap = torch.sigmoid(z_c_cap)
                    output_sm_c = torch.sigmoid(z_c)            
                    
                    ##### CROCODILE block (dataset/domain prediction)######  
                    if args.useCrocodile:
                        loss1_crocodile = criterion_crocodile(z_x_crocodile, target_crocodile) #the supervised loss
                        if torch.isnan(loss1_crocodile).any():                    
                            logger.info("VALID Loss - loss1_crocodile has NaNs: raise ValueError and exit")
                            raise ValueError                
                        uniform_target_crocodile = torch.ones_like(z_c_crocodile, dtype=torch.float, device='cuda') / args.num_class_crocodile #we need to push its prediction equally to all categories
                        loss2_crocodile = F.kl_div(F.log_softmax(z_c_crocodile, dim=1), uniform_target_crocodile, reduction='batchmean')
                        if torch.isnan(loss2_crocodile).any():
                            logger.info("VALID Loss - loss2_crocodile has NaNs: raise ValueError and exit")
                            raise ValueError                
                        loss3_crocodile = criterion_crocodile(z_c_cap_crocodile, target_crocodile)
                        if torch.isnan(loss3_crocodile).any():
                            logger.info("VALID Loss - loss3_crocodile has NaNs: raise ValueError and exit")
                            raise ValueError                
                        
                        loss_crocodile = args.w_CE_d*loss1_crocodile + args.w_KL_d*loss2_crocodile + args.w_CE_d_bd*loss3_crocodile
                        
                        # compute the probabilities from the classification logits
                        output_sm_x_crocodile = torch.softmax(z_x_crocodile, dim=1)
                        output_sm_c_cap_crocodile = torch.softmax(z_c_cap_crocodile, dim=1)
                        output_sm_c_crocodile = torch.softmax(z_c_crocodile, dim=1)   

                        if args.useContrastiveLoss:
                            loss_minibatchLevel = args.w_Yca_sameY*loss_Y_ca_sameY + args.w_Ysp_diffY*loss_Y_sp_differentY + args.w_Dca_sameD*loss_D_ca_sameD + args.w_Dsp_diffD*loss_D_sp_differentD

                            if args.contrastiveLossSpace=="activation": # this option is not fully tested yet, not used in the paper. Please, use the "representation" option.
                                #Differently to the 'representation' case, which already have a 'loss_imageLevel', here, we create this new variable by unifying the two terms and give it the same name
                                loss_imageLevel = args.w_Y_imgLvl*loss_Y_ca_D_sp + args.w_Y_imgLvl*loss_Y_sp_D_ca 

                    ## Now compose the total loss. Depending on the args of the experiment, some terms might not be set (value 0):
                    loss = loss_caumed + loss_crocodile + loss_imageLevel + loss_minibatchLevel + loss_task_prior

                    if dist.get_rank() == 0:
                        if summary_writer:
                            # tensorboard logger
                            summary_writer.add_scalar('ValidLoss', loss, i)
                            summary_writer.add_scalar('ValidLoss/CauMed', loss_caumed, i)
                            summary_writer.add_scalar('ValidLoss/CauMed/CE_y', args.w_CE_y*loss1, i)
                            summary_writer.add_scalar('ValidLoss/CauMed/KL_y', args.w_KL_y*loss2, i)
                            summary_writer.add_scalar('ValidLoss/CauMed/CE_bd_y', args.w_CE_y_bd*loss3, i)
                            summary_writer.add_scalar('ValidLoss/Domain', loss_crocodile, i)
                            summary_writer.add_scalar('ValidLoss/Domain/CE_d', args.w_CE_d*loss1_crocodile, i)
                            summary_writer.add_scalar('ValidLoss/Domain/KL_d', args.w_KL_d*loss2_crocodile, i)
                            summary_writer.add_scalar('ValidLoss/Domain/CE_bd_d', args.w_CE_d_bd*loss3_crocodile, i)
                            summary_writer.add_scalar('ValidLoss/ContrImageLevel', args.w_Y_imgLvl*loss_imageLevel, i)
                            summary_writer.add_scalar('ValidLoss/ContrBatchLevel', loss_minibatchLevel, i)
                            summary_writer.add_scalar('ValidLoss/ContrBatchLevel/Y_ca_sameY', args.w_Yca_sameY*loss_Y_ca_sameY, i)
                            summary_writer.add_scalar('ValidLoss/ContrBatchLevel/Y_sp_differentY', args.w_Ysp_diffY*loss_Y_sp_differentY, i)
                            summary_writer.add_scalar('ValidLoss/ContrBatchLevel/D_ca_sameD', args.w_Dca_sameD*loss_D_ca_sameD, i)
                            summary_writer.add_scalar('ValidLoss/ContrBatchLevel/D_sp_differentD', args.w_Dsp_diffD*loss_D_sp_differentD, i)
                            summary_writer.add_scalar('ValidLoss/TaskPrior', args.w_Yprior*loss_task_prior, i)
                        if random.random()<0.1: # evry so often, print the loss values
                            logger.info(f"|\tVALID LOSS: {loss}\n\
                                        \tCAU-MED:\t{loss_caumed}\n\
                                        \t\targs.w_CE_y*loss1:\t{args.w_CE_y}*{loss1}=\t{args.w_CE_y*loss1}\n\
                                        \t\targs.w_KL_y*loss2:\t{args.w_KL_y}*{loss2}=\t{args.w_KL_y*loss2}\n\
                                        \t\targs.w_CE_y_bd*loss3:\t{args.w_CE_y_bd}*{loss3}=\t{args.w_CE_y_bd*loss3}\n\
                                        \tDOMAIN/DATASET:\t{loss_crocodile}\n\
                                        \t\targs.w_CE_d*loss1_crocodile:\t{args.w_CE_d}*{loss1_crocodile}=\t{args.w_CE_d*loss1_crocodile}\n\
                                        \t\targs.w_KL_d*loss2_crocodile:\t{args.w_KL_d}*{loss2_crocodile}=\t{args.w_KL_d*loss2_crocodile}\n\
                                        \t\targs.w_CE_d_bd*loss3_crocodile:\t{args.w_CE_d_bd}*{loss3_crocodile}=\t{args.w_CE_d_bd*loss3_crocodile}\n\
                                        \tCONTR-ImageLevel:\t{loss_imageLevel}\n\
                                        \t\targs.w_Y_imgLvl:\t{args.w_Y_imgLvl}\n\
                                        \tCONTR-MinibatchLevel:\t{loss_minibatchLevel}\n\
                                        \t\targs.w_Yca_sameY*loss_Y_ca_sameY:\t{args.w_Yca_sameY}*{loss_Y_ca_sameY}=\t{args.w_Yca_sameY*loss_Y_ca_sameY}\n\
                                        \t\targs.w_Ysp_diffY*loss_Y_sp_differentY:\t{args.w_Ysp_diffY}*{loss_Y_sp_differentY}=\t{args.w_Ysp_diffY*loss_Y_sp_differentY}\n\
                                        \t\targs.w_Dca_sameD*loss_D_ca_sameD:\t{args.w_Dca_sameD}*{loss_D_ca_sameD}=\t{args.w_Dca_sameD*loss_D_ca_sameD}\n\
                                        \t\targs.w_Dsp_diffD*loss_D_sp_differentD:\t{args.w_Dsp_diffD}*{loss_D_sp_differentD}=\t{args.w_Dsp_diffD*loss_D_sp_differentD}\n\
                                        \tTASK-Prior:\t{loss_task_prior}\n\
                                        \t\targs.w_Yprior:\t{args.w_Yprior}")
                                
                    if args.loss_dev > 0:
                        loss *= args.loss_dev         
                        
                losses.update(loss* args.batch_size/world_size, images.size(0))

                tar = target.cpu()            
                out_x = output_sm_x.cpu()
                out_c_cap = output_sm_c_cap.cpu()
                out_c = output_sm_c.cpu()
                targets = tar if targets == None else torch.cat([targets, tar])
                outputs_x = out_x if outputs_x == None else torch.cat([outputs_x, out_x])
                outputs_c_cap = out_c_cap if outputs_c_cap == None else torch.cat([outputs_c_cap, out_c_cap])
                outputs_c = out_c if outputs_c == None else torch.cat([outputs_c, out_c])                       

                

                # # save some data                
                _item_x = torch.cat((out_x, tar), 1)
                _item_c_cap = torch.cat((out_c_cap, tar), 1)
                _item_c = torch.cat((out_c, tar), 1)

                saved_data_x.append(_item_x)
                saved_data_c_cap.append(_item_c_cap)
                saved_data_c.append(_item_c)

                if args.useCrocodile: 
                    tar_crocodile = target_crocodile.cpu()
                    out_x_crocodile = output_sm_x_crocodile.cpu()
                    out_c_cap_crocodile = output_sm_c_cap_crocodile.cpu()
                    out_c_crocodile = output_sm_c_crocodile.cpu()
                    targets_crocodile = tar_crocodile if targets_crocodile == None else torch.cat([targets_crocodile, tar_crocodile])
                    outputs_x_crocodile = out_x_crocodile if outputs_x_crocodile == None else torch.cat([outputs_x_crocodile, out_x_crocodile])
                    outputs_c_cap_crocodile = out_c_cap_crocodile if outputs_c_cap_crocodile == None else torch.cat([outputs_c_cap_crocodile, out_c_cap_crocodile])
                    outputs_c_crocodile = out_c_crocodile if outputs_c_crocodile == None else torch.cat([outputs_c_crocodile, out_c_crocodile])

                mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) # in megabytes

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0 and dist.get_rank() == 0:
                    progress.display(i, logger)

        logger.info('=> synchronize...') # synchronize all processes of the DDP multithreading thanks to the barrier method
        if dist.get_world_size() > 1:
            dist.barrier()
        
        if not args.evaluate: # training and validation setting
            loss_avg, = map(
                _meter_reduce if dist.get_world_size() > 1 else lambda x: x.avg,
                [losses]
            )
        else:  # test setting (since during testset evaluation we do not track the loss)
            loss_avg = 0

        ## calculate mAP
        saved_data_x = torch.cat(saved_data_x, 0).numpy()
        saved_data_c_cap = torch.cat(saved_data_c_cap, 0).numpy()
        saved_data_c = torch.cat(saved_data_c, 0).numpy()

        saved_name_x = f'saved_data_x_{dist.get_rank()}.txt'
        saved_name_c_cap = f'saved_data_c_cap_{dist.get_rank()}.txt'
        saved_name_c = f'saved_data_c_{dist.get_rank()}.txt'

        np.savetxt(os.path.join(args.output, saved_name_x), saved_data_x)
        np.savetxt(os.path.join(args.output, saved_name_c_cap), saved_data_c_cap)
        np.savetxt(os.path.join(args.output, saved_name_c), saved_data_c)

        if dist.get_world_size() > 1:
            dist.barrier()

        if dist.get_rank() == 0:
            
            filenamelist_x = ['saved_data_x_{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            filenamelist_c_cap = ['saved_data_c_cap_{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            filenamelist_c = ['saved_data_c_{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            
            metric_func = voc_mAP
            print("Calculating mAP:")
            mAP_x, aps_x = metric_func([os.path.join(args.output, _filename) for _filename in filenamelist_x], args.num_class,return_each=True)
            mAP_c_cap, aps_c_cap = metric_func([os.path.join(args.output, _filename) for _filename in filenamelist_c_cap], args.num_class,return_each=True)
            mAP_c, aps_c = metric_func([os.path.join(args.output, _filename) for _filename in filenamelist_c], args.num_class,return_each=True)
            
            logger.info("  meanAP (x): {}".format(np.array2string(mAP_x, precision=3)))
            logger.info("     aps (x): {}".format(np.array2string(aps_x, precision=4)))
            logger.info("  meanAP (c_cap): {}".format(np.array2string(mAP_c_cap, precision=3)))
            logger.info("     aps (c_cap): {}".format(np.array2string(aps_c_cap, precision=4)))
            logger.info("  meanAP (c): {}".format(np.array2string(mAP_c,precision=3)))
            logger.info("     aps (c): {}".format(np.array2string(aps_c, precision=4)))
        else:
            mAP_x=0
            mAP_c_cap=0
            mAP_c=0
            aps_x = 0
            aps_c_cap = 0
            aps_c = 0

        if dist.get_world_size() > 1:
            dist.barrier()
        

    ##### Compute the AUC scores and mean-AUC score for the different classifiers
    outputs_x = outputs_x.detach().numpy()
    outputs_c_cap = outputs_c_cap.detach().numpy()
    outputs_c = outputs_c.detach().numpy()
    targets = targets.detach().numpy()

    auc_scores_x = roc_auc_score(targets, outputs_x, average=None) # get the AUROC score for each class
    auc_scores_c_cap = roc_auc_score(targets, outputs_c_cap, average=None)
    auc_scores_c = roc_auc_score(targets, outputs_c, average=None)

    NOFINDING_position_in_list=cate.index('No Finding')
    meanAUC_x = (sum(auc_scores_x)-auc_scores_x[NOFINDING_position_in_list]) / (args.num_class-1) # We can automatically exclude the No Finding class from the mean calculation, if needed
    meanAUC_c_cap = (sum(auc_scores_c_cap)-auc_scores_c_cap[NOFINDING_position_in_list]) / (args.num_class-1)
    meanAUC_c = (sum(auc_scores_c)-auc_scores_c[NOFINDING_position_in_list]) / (args.num_class-1)

    logger.info("AUROC scores (x): {}".format(auc_scores_x))
    logger.info("mean AUROC (x): {}".format(meanAUC_x))
    logger.info("AUROC scores (c_cap): {}".format(auc_scores_c_cap))
    logger.info("mean AUROC (c_cap): {}".format(meanAUC_c_cap))
    logger.info("AUROC scores (c): {}".format(auc_scores_c))
    logger.info("mean AUROC (c): {}".format(meanAUC_c))

    if args.useCrocodile:
        outputs_x_crocodile = outputs_x_crocodile.detach().numpy()
        outputs_c_cap_crocodile = outputs_c_cap_crocodile.detach().numpy()
        outputs_c_crocodile = outputs_c_crocodile.detach().numpy()
        targets_crocodile = targets_crocodile.detach().numpy()
        
        #### Compute the single scores for each dataset/domain class: in this case we have a MULTI-CLASS setting
        ## Actually, we can comment the following lines and go to the mean calculation directly, as we do not need to print the single scores for each class of the domain-prediction branch.
        # auc_scores_x_crocodile = roc_auc_score(targets_crocodile, outputs_x_crocodile, average=None, multi_class='ovo')
        # auc_scores_c_cap_crocodile = roc_auc_score(targets_crocodile, outputs_c_cap_crocodile, average=None, multi_class='ovo')
        # auc_scores_c_crocodile = roc_auc_score(targets_crocodile, outputs_c_crocodile, average=None, multi_class='ovo')
        # logger.info("AUROC scores domain/dataset task (x_crocodile): {}".format(auc_scores_x_crocodile))
        # logger.info("AUROC scores domain/dataset task (c_cap_crocodile): {}".format(auc_scores_c_cap_crocodile))
        # logger.info("AUROC scores domain/dataset task (c_crocodile): {}".format(auc_scores_c_crocodile))
        #### Compute their average (in this case, we do not have a class to be excluded such as No Finding, so we use the builtin 'average' argument to achieve the mean count)
        meanAUC_x_crocodile = roc_auc_score(targets_crocodile, outputs_x_crocodile, average='macro',multi_class='ovo') # we utilize the one-vs-one strategy for multi-class problems
        meanAUC_c_cap_crocodile = roc_auc_score(targets_crocodile, outputs_c_cap_crocodile, average='macro',multi_class='ovo')
        meanAUC_c_crocodile =  roc_auc_score(targets_crocodile, outputs_c_crocodile, average='macro',multi_class='ovo')
        logger.info("mean AUROC domain/dataset task (x_crocodile): {}".format(meanAUC_x_crocodile))
        logger.info("mean AUROC domain/dataset task (c_cap_crocodile): {}".format(meanAUC_c_cap_crocodile))
        logger.info("mean AUROC domain/dataset task (c_crocodile): {}".format(meanAUC_c_crocodile))

    return loss_avg, mAP_x, meanAUC_x, mAP_c_cap, meanAUC_c_cap, mAP_c, meanAUC_c, Q, Q_bar, Q_crocodile, Q_bar_crocodile, cmap_Q, aps_x, aps_c_cap, aps_c, auc_scores_x, auc_scores_c_cap, auc_scores_c


######## UTILITY FUNCTIONS AND CLASSES  ##########################################################################
def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def _meter_reduce(meter):
    meter_sum = torch.Tensor([meter.sum]).cuda()
    meter_count = torch.Tensor([meter.count]).cuda()

    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, os.path.split(filename)[0] + '/best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class AverageMeterHMS(AverageMeter):
    """Meter for timer in HH:MM:SS format"""

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val}'
        else:
            fmtstr = '{name} {val} ({sum})'
        return fmtstr.format(name=self.name,
                             val=str(datetime.timedelta(seconds=int(self.val))),
                             sum=str(datetime.timedelta(seconds=int(self.sum))))

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def kill_process(filename: str, holdpid: int) -> List[str]:
    import subprocess, signal
    res = subprocess.check_output("ps aux | grep {} | grep -v grep | awk '{{print $2}}'".format(filename), shell=True,
                                  cwd="./")
    res = res.decode('utf-8')
    idlist = [i.strip() for i in res.split('\n') if i != '']
    print("kill: {}".format(idlist))
    for idname in idlist:
        if idname != str(holdpid):
            os.kill(int(idname), signal.SIGKILL)
    return idlist

def voc_ap(rec, prec, true_num):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_mAP(imagessetfilelist, num, return_each=False):
    '''
    Compute the Average Precision Scores (aps) that summarize the Precision-Recall Curve
    Also compute the mean average precision (mAP) across all the categories (classes).

    "Much like ROC curves, we can summarize the information in a precision-recall curve with a single value.
    This summary metric is the AUC-PR. AUC-PR stands for area under the (precision-recall) curve.
    Generally, the higher the AUC-PR score, the better a classifier performs for the given task.
    One way to calculate AUC-PR is to find the AP, or average precision.
    The documentation for sklearn.metrics.average_precision_score states:
    AP summarizes a precision-recall curve as the weighted mean of precision achieved at each threshold,
    with the increase in recall from the previous threshold used as the weight.
    So, we can think of AP as a kind of weighted-average precision across all thresholds."

    Precision-Recall is a useful measure of success of prediction when the classes are very imbalanced.
    The precision-recall curve shows the tradeoff between precision and recall for different threshold.
    A high area under the curve represents both high recall and high precision, where
    high precision relates to a low false positive rate, and high recall relates to a low false negative rate.
    High scores for both show that the classifier is returning accurate results (high precision),
    as well as returning a majority of all positive results (high recall).
    '''
    if isinstance(imagessetfilelist, str):
        imagessetfilelist = [imagessetfilelist]
    lines = []
    for imagessetfile in imagessetfilelist:
        with open(imagessetfile, 'r') as f:
            lines.extend(f.readlines())

    seg = np.array([x.strip().split(' ') for x in lines]).astype(float)
    del lines
    gt_label = seg[:, num:].astype(np.int32)
    num_target = np.sum(gt_label, axis=1, keepdims=True)
    sample_num = len(gt_label)
    class_num = num
    tp = np.zeros(sample_num)
    fp = np.zeros(sample_num)
    aps = []

    for class_id in range(class_num):
        confidence = seg[:, class_id]
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        sorted_label = [gt_label[x][class_id] for x in sorted_ind]

        for i in range(sample_num):
            tp[i] = (sorted_label[i] > 0)
            fp[i] = (sorted_label[i] <= 0)
        true_num = 0
        true_num = sum(tp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(true_num)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, true_num)
        aps += [ap]

    np.set_printoptions(precision=6, suppress=True)
    aps = np.array(aps) * 100
    mAP = np.mean(aps)
    if return_each:
        return mAP, aps
    return mAP

def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict
###################################################################

if __name__ == '__main__':
    print("__main__ begins") 
    args = get_args()
    # Set the random seed for reproducibility if provided
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    ## To train with a single computing node (with possibly multiple GPUs), such as the Docker-based NVIDIA DGX cluster at ISTI-CNR Institute:
    # you shuld retrieve the world_size from the args, and then use the multiprocessing.spawn method specifying the nprocs=world_size
    #  E.g, something like:
    #       world_size=args.number_of_gpus
    #       mp.spawn(main, args=(world_size,args), nprocs=world_size)
    
    ## Instead, we train with a multi-node multi-gpu cluster, such as the LEONARDO Supercomputer (Bologna, Italy):
    # In this case, we use the SLURM environment variables to set the rank and world_size, and then simply call the main function
    # (This actually applies also to a setting where you have just one node with a single GPU: it would assign world_size=1 and rank=0, and the DDP collapes to a regular un-parallelized training)
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])

    main(rank=rank,
         world_size=world_size,
         args=args
    )