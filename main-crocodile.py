'''
Late April 2024: we try to implement what we call CROCODILE. Here, multiple datasets are used in a contrastive learning setting to attain robustness.
In Leonardo SCRATCH, we have: dataset_chestxray14(NIH), dataset_chexpert, dataset_mimimCXR_JPG, and dataset_padchest

https://github.com/gianlucarloni/crocodile

'''
import argparse
import math
import os, sys
#Matplotlib created a temporary config/cache directory at /tmp/matplotlib-6772vh08 because the default path (/.config/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

import _init_paths 

import random
import datetime
import time
from typing import List
import json
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm

from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import torch 
##
if torch.cuda.is_available():
    print(f"TORCH Number of available GPUs: {torch.cuda.device_count()}")
##
import torch.nn as nn
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# from torch.profiler import profile, record_function, ProfilerActivity

# import _init_paths
## Utilize the lib/dataset lib/models and lib/utils subfolders of the project
from dataset.get_dataset import get_datasets
from utils.logger import setup_logger
import models
import models.aslloss
from models.net import build_net
from collections import OrderedDict
from utils.config_ import get_raw_dict

#
from dataset.cxr_datasets import cate #radiological findings categories, ie, the classes
# import hostlist
#
torch.autograd.set_detect_anomaly(True) #TODO set this to true, for debugging purposes, such as checking expoding/vanishing gradients, or NaN values within Tensors...
#
mydir=os.path.join(os.getcwd(), 'pretrained_models') ## these are the regular ImageNet-trained CV models, like resnets weights.
torch.hub.set_dir(mydir)
os.environ['TORCH_HOME']=mydir


#TODO in multi-node multi gpu with slurm, we do not need to set this explicitly
##### To implement the DistributedDataParallel solution in PyTorch, it is necessary to:
##Define the environment variables linked to the master node
# os.environ['MASTER_ADDR'] = 'localhost' #The IP address or the hostname of the node corresponding to task 0 (the first node on the node list). If you are in mono-node, the value localhost is sufficient.
# os.environ['MASTER_PORT'] = '12355' #The number of a random port. To avoid conflicts, and by convention, we will use a port number between 10001 and 20000, such as 12355, 6668

# ### Otherwise, we can make it automatic with the following code snippet:
# hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST']) ## get node list from slurm
# gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",") ## get IDs of reserved GPU
# # define MASTER_ADD & MASTER_PORT
# os.environ['MASTER_ADDR'] = hostnames[0]
# os.environ['MASTER_PORT'] = str(12345 + int(min(gpu_ids))) # to avoid port conflict on the same node
# #####



# default_collate_func = torch.utils.data.dataloader.default_collate

def parser_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument("number_of_gpus",type=int,help="The number of GPUs you intend to use")
    parser.add_argument('--useCrocodile', action='store_true',
                        help='Utilize the CROCODILE setting. default is False. ')
    parser.add_argument('--note', help='note', default='Causal experiment')
    parser.add_argument('--dataname', help='dataname', default='nih')
    parser.add_argument('--kl', help='kl loss *', default=0.55) #TODO alpha_1 weight of Eq.8, not 1 but 0.55
    parser.add_argument('--ce2', help='half loss *', default=0.25) #TODO alpha_2 weight in eq.8, not 0.5, but 0.45
    parser.add_argument('--dataset_dir', help='dir of dataset', default='./data')
    parser.add_argument('--img_size', default=224, type=int,
                        help='size of input images')
    parser.add_argument('--output', default='./out/baseline/{}'.format( #TODO
        # time.strftime("%Y%m%d%H%M%S", time.localtime(time.time() + 28800))), metavar='DIR',
        time.strftime("%Y%m%d%H%M", time.localtime(time.time() + 7200))), metavar='DIR',
                        help='path to output folder')
    parser.add_argument('--num_class', default=9, type=int, #TODO non 15 ma 9 le comuni ora
                        help="Number of query slots")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False. ')
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'],
                        help='which optim to use')
    
    #TODO
    parser.add_argument('--dropoutrate_randomaddition', default=0.35, type=float,
                        help="Dropout probability to zero out elements in the Confounding feature set to be randomly added to the causal one")
    #TODO
    parser.add_argument('--subset_take1everyN', default=2, type=int)


    # loss
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
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    # parser.add_argument('--ema-decay', default=0.9997, type=float, metavar='M',
    #                     help='decay of model ema')
    # parser.add_argument('--ema-epoch', default=0, type=int, metavar='M',
    #                     help='start ema epoch')

    # distribution training
    parser.add_argument('--world_size', type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    # parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    # data aug
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

    # * Transformer
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

    # * Training
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
    return args


# best_mAP = 0
# best_meanAUC = 0
best_mAP_c_cap = 0 #TODO
best_meanAUC_c_cap = 0

## TODO May 2024: commento questa porzione e la porto dentro il __main__ in fondo (ovviamente ora la funzione def main() ha bisogno dell argomento args)
# ## TODO 12 ottobre nuovo, con distributed training
# args = get_args()
# NUM_GPUS = args.number_of_gpus 
# # list_of_GPU_ids = list(args.gpus_ids)
# # list_of_GPU_ids = list(filter((",").__ne__, list_of_GPU_ids))

# if args.seed is not None:
#     random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)

# def main(rank, world_size):
def main(rank, world_size, args): #TODO May 2024

    ## os.environ['CUDA_VISIBLE_DEVICES'] = list_of_GPU_ids[rank]
    # args.world_size = world_size
    # args.local_rank = rank
    ##

    # if 'WORLD_SIZE' in os.environ:
    #     print("MAIN - WORLDSIZE detected...")
    #     assert args.world_size > 0, 'please set --world-size and --rank in the command line'

    #     local_world_size = int(os.environ['WORLD_SIZE'])
    #     args.world_size = args.world_size * local_world_size
    #     args.rank = args.rank * local_world_size + args.local_rank
    #     print('world size: {}, world rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
    #     print('os.environ:', os.environ)
    # else:
    #     # single process, useful for debugging
    #     #   python main.py ...
    #     print("MAIN - WORLDSIZE=1, single process with rank=0 and local_rank=0.")
    #     args.world_size = 1
    #     args.rank = 0
    #     args.local_rank = 0

    # torch.cuda.set_device(rank) #TODO in multi-node multi gpu with slurm, we do not need to set this explicitly

    # print('| distributed init (local_rank {}): {}'.format(rank, args.dist_url), flush=True)
    print(f'MAIN | Distributed init, local_rank {rank} (worldsize: {world_size}), CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}, is CUDA available:{torch.cuda.is_available()}): {args.dist_url}')

    # os.environ['MASTER_ADDR'] = 'localhost' #TODO in multi-node multi gpu with slurm, we do not need to set this explicitly
    # os.environ['MASTER_PORT'] = '12355' #TODO in multi-node multi gpu with slurm, we do not need to set this explicitly
    # print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
   

    torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url,
                                         world_size=world_size, rank=rank)
        
    cudnn.benchmark = True

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="CROCODILE")

    
    logger.info("Command: " + ' '.join(sys.argv))
    if dist.get_rank() == 0:
        path = os.path.join(args.output, "config.json")
        with open(path, 'w') as f:
            json.dump(get_raw_dict(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))

        logger.info('world size: {}'.format(dist.get_world_size()))
        logger.info('dist.get_rank(): {}'.format(dist.get_rank()))
        logger.info('local_rank: {}'.format(rank))

    return main_worker(rank, world_size, args, logger)


def main_worker(rank, world_size, args, logger):
    print("Main worker")
    # global best_mAP
    # global best_meanAUC
    global best_mAP_c_cap
    global best_meanAUC_c_cap


    # build model
    model = build_net(args,logger)#TODO May 2024, added the logger object
    model = model.cuda()
    # print(f"main_worker(args, logger) - MODEL DEVICE is CUDA?: {next(model.parameters()).is_cuda}")

    ## Model Exponential Moving Average (EMA):
    # Empirically it has been found that using the moving average of the trained parameters
    # of a deep network is better than using its trained parameters directly.
    # Indeed, When training a model, it is often beneficial to maintain moving averages
    # of the trained parameters. Evaluations that use averaged parameters sometimes produce
    # significantly better results than the final trained values:
    # ema_m = ModelEma(model, args.ema_decay)  # 0.9997


    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=False, find_unused_parameters=True)
    # model = torch.nn.DataParallel(model)

    # criterion
    criterion = models.aslloss.AsymmetricLossOptimized(
        gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
        disable_torch_grad_focal_loss=args.dtgfl,
        eps=args.eps,
    )

    # optimizer
    # args.lr_mult = args.batch_size / 256
    args.lr_mult = 1 #TODO

    if args.optim == 'AdamW':
        param_dicts = [
            {"params": [p for n, p in model.module.named_parameters() if p.requires_grad]},
        ]
        optimizer = getattr(torch.optim, args.optim)(
            param_dicts,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )
    elif args.optim == 'Adam_twd': #Without weight-decay
        print("Using Adam_twd, so without the weight decay")
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(
            parameters,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )
    else:
        raise NotImplementedError

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output)
    else:
        summary_writer = None
        # summary_writer = SummaryWriter(log_dir=args.output)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(dist.get_rank()))

            if 'state_dict' in checkpoint:
                state_dict = clean_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                state_dict = clean_state_dict(checkpoint['model'])
            else:
                logger.info("No model or state_dict Found!!!")
                raise ValueError("No model or state_dict Found!!!")
            logger.info("Omitting {}".format(args.resume_omit))
            # import ipdb; ipdb.set_trace()
            for omit_name in args.resume_omit:
                del state_dict[omit_name]
            model.module.load_state_dict(state_dict, strict=False)
            # model.module.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
            del checkpoint
            del state_dict
            torch.cuda.empty_cache() #TODO 20 oct: commented out this line because it slows the process...
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    print(f"Get datasets...with args.useCrocodile={args.useCrocodile}")    
    train_dataset, val_dataset = get_datasets(args)

    

    

    ## TODO for debugging purposes: take a subset of the whole dataset to arrive at some conclusion faster
    # Attention: that could yield some problems with the AUROC computing in the case the few samples considere are all of the same class.
    subset_interval=args.subset_take1everyN #take one every N samples

    if subset_interval != 1: # 1 is the normal condition mode, take every image in the dataset
        idx_trainval = list(range(0, len(train_dataset), subset_interval))
        idx_test = list(range(0, len(val_dataset), subset_interval))
        train_dataset = torch.utils.data.Subset(train_dataset, idx_trainval)
        val_dataset = torch.utils.data.Subset(val_dataset, idx_test)
        if dist.get_rank() == 0:
            logger.info('DEBUG-using subset of datasets, 1 every {} samples'.format(subset_interval))
        ####
    











    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,shuffle=True,drop_last=False)
    
    assert args.batch_size % dist.get_world_size() == 0 #'Batch size is not divisible by num of gpus.' #TODO commented out
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= int(args.batch_size / dist.get_world_size()), shuffle=False,pin_memory=True, num_workers=args.workers, sampler=train_sampler, drop_last=True)
    # print(f"dataloader_train of size {len(train_loader)} batches, each of {args.batch_size}")

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(args.batch_size / dist.get_world_size()), shuffle=False,pin_memory=True, num_workers=args.workers, sampler=val_sampler, drop_last=True)

    if args.evaluate:
        # _, mAP = validate(val_loader, model, criterion, args, logger)
        # logger.info(' * mAP {mAP:.5f}'
        #             .format(mAP=mAP))
        _, mAP_x, _, mAP_c_cap, _, mAP_c, _ = validate(val_loader, model, criterion, args, logger)
        logger.info(f"mAP (x) {mAP_x}")
        logger.info(f"mAP (c_cap) {mAP_c_cap}")
        logger.info(f"mAP (c) {mAP_c}")        
        return

    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    # losses_ema = AverageMeter('Loss_ema', ':5.3f', val_only=True)
    ## mAPs = AverageMeter('mAP', ':5.5f', val_only=True)
    mAPs_x = AverageMeter('mAP_x', ':5.5f', val_only=True)
    mAPs_c_cap = AverageMeter('mAP_c_cap', ':5.5f', val_only=True)
    mAPs_c = AverageMeter('mAP_c', ':5.5f', val_only=True)

    ## mAPs_ema = AverageMeter('mAP_ema', ':5.5f', val_only=True)
    # mAPs_x_ema = AverageMeter('mAP_x_ema', ':5.5f', val_only=True)
    # mAPs_c_cap_ema = AverageMeter('mAP_c_cap_ema', ':5.5f', val_only=True)
    # mAPs_c_ema = AverageMeter('mAP_c_ema', ':5.5f', val_only=True)

    progress = ProgressMeter(
        args.epochs,
        # [eta, epoch_time, losses, mAPs, losses_ema, mAPs_ema],
        [eta, epoch_time, losses, mAPs_x, mAPs_c_cap,mAPs_c],# losses_ema, mAPs_x_ema, mAPs_c_cap_ema, mAPs_c_ema],
        prefix='=> Test Epoch: ')

    # one cycle learning rate
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                        epochs=args.epochs, pct_start=0.2)
    ## TODO changed the scheduler to MultiStep, with milestones at 1/4, 2/4 and 3/4 of the epochs.
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(0.25*args.epochs), round(0.50*args.epochs), round(0.75*args.epochs),], gamma=0.5)
    # TODO 15 Jan 2024: the golden ratio series within [0-40] epoch range.
    # milestones_golden=[1,2,3,5,8,13,21,34]
    # milestones_golden=[1,2,4,6,10,16,26] #are the rounded integer for the golden series witin number 42.    
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones_golden, gamma=0.61803) #1/1.61803 Golder Ratio
    # TODO 27 Jan 2024:
    # milestones_golden=[6,16,26] #are the rounded integer for the golden series witin number 42.    
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones_golden, gamma=0.61803) #1/1.61803 Golder Ratio


    end = time.time()
    # best_regular_meanAUC = 0
    # best_ema_meanAUC = 0
    # best_epoch = -1
    # best_regular_mAP = 0
    # best_regular_epoch = -1
    # best_ema_mAP = 0
    # regular_mAP_list = []
    # ema_mAP_list = []
    #TODO versione mia
    best_regular_meanAUC_c_cap = 0
    # best_ema_meanAUC_c_cap = 0
    best_epoch_c_cap = -1
    # best_regular_mAP = 0
    best_regular_epoch_c_cap = -1
    # best_ema_mAP_c_cap = 0
    regular_mAP_c_cap_list = regular_mAP_c_list = regular_mAP_x_list =[]
    # ema_mAP_c_cap_list = ema_mAP_c_list = ema_mAP_x_list = []

    torch.cuda.empty_cache() #TODO 20 oct: commented out this line because it slows the process...

    ##
    print(f"Loss is made of three terms: Supervised loss, confounding loss, and backdoor loss\n Loss = Loss_sl + {args.kl}*Loss_conf + {args.ce2}*Loss_bd")
    ##


    #TODO 12 Feb 2024 using Accelerator:
    # train_loader, val_loader, model, criterion, optimizer, scheduler = accelerator.prepare(train_loader, val_loader, model, criterion, optimizer, scheduler)
    ##



    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        # if args.ema_epoch == epoch:
        #     ema_m = ModelEma(model.module, args.ema_decay)
        #     # torch.cuda.empty_cache() #TODO 20 oct: commented out this line because it slows the process...

        #     #TODO 12 Feb 2024 using Accelerator:
        #     ema_m = accelerator.prepare(ema_m)
        #     ##

        torch.cuda.empty_cache() #TODO 20 oct: commented out this line because it slows the process...

        startt = time.time()

        # train for one epoch

        #TODO 20 oct: inserisco un Profiler
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,record_shapes=True) as prof_train:
        #     with record_function("TRAIN_model"):
        #         loss = train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger)
        # print(prof_train.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=15))

        # loss = train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger) #TODO
        loss = train(train_loader, model, None, criterion, optimizer, scheduler, epoch, args, logger)


        # ##
        # #TODO moved the scheduler.step procedure from within the train function to outside it
        # if epoch in [0, round(0.25*args.epochs), round(0.50*args.epochs), round(0.75*args.epochs)]:
        #     before_lr = optimizer.param_groups[0]["lr"]
        #     scheduler.step()
        #     after_lr = optimizer.param_groups[0]["lr"]
        #     print(f"Epoch {epoch}: lr {before_lr}--->{after_lr}")
        # else:#update anyway but does not show any message
        #     scheduler.step()
        # ##

        endt = time.time()
        logger.info("Elapsed time:    {} hours".format((endt - startt)/3600))

        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            # summary_writer.add_scalar('train_acc1', acc1, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % args.val_interval == 0:

            ## evaluate on validation set
            # loss, mAP, meanAUC = validate(val_loader, model, criterion, args, logger, epoch)
            # loss_ema, mAP_ema, meanAUC_ema = validate(val_loader, ema_m.module, criterion, args, logger, epoch)
            loss, mAP_x, meanAUC_x, mAP_c_cap, meanAUC_c_cap, mAP_c, meanAUC_c= validate(val_loader, model, criterion, args, logger, epoch)
            # loss_ema, mAP_x_ema, meanAUC_x_ema, mAP_c_cap_ema, meanAUC_c_cap_ema, mAP_c_ema, meanAUC_c_ema= validate(val_loader, ema_m.module, criterion, args, logger, epoch)
            
            losses.update(loss)
            # mAPs.update(mAP)
            mAPs_x.update(mAP_x)
            mAPs_c_cap.update(mAP_c_cap)
            mAPs_c.update(mAP_c)

            # losses_ema.update(loss_ema)
            # mAPs_ema.update(mAP_ema)
            # mAPs_x_ema.update(mAP_x_ema)
            # mAPs_c_cap_ema.update(mAP_c_cap_ema)
            # mAPs_c_ema.update(mAP_c_ema)


            epoch_time.update(time.time() - end)
            end = time.time()
            eta.update(epoch_time.avg * (args.epochs - epoch - 1))

            ## regular_mAP_list.append(mAP)
            ## ema_mAP_list.append(mAP_ema)
            # regular_mAP_x_list.append(mAP_x)
            # ema_mAP_x_list.append(mAP_x_ema)
            # regular_mAP_c_cap_list.append(mAP_c_cap)
            # ema_mAP_c_cap_list.append(mAP_c_cap_ema) # it is used for early stop triggering
            # regular_mAP_c_list.append(mAP_c)
            # ema_mAP_c_list.append(mAP_c_ema)

            progress.display(epoch, logger)

            if summary_writer:
                # tensorboard logger
                summary_writer.add_scalar('val_loss', loss, epoch)
                # summary_writer.add_scalar('val_mAP', mAP, epoch)
                summary_writer.add_scalar('val_mAP_x', mAP_x, epoch)
                summary_writer.add_scalar('val_mAP_c_cap', mAP_c_cap, epoch)
                summary_writer.add_scalar('val_mAP_c', mAP_c, epoch)

                # summary_writer.add_scalar('val_loss_ema', loss_ema, epoch)
                # summary_writer.add_scalar('val_mAP_ema', mAP_ema, epoch)
                # summary_writer.add_scalar('val_mAP_x_ema', mAP_x_ema, epoch)
                # summary_writer.add_scalar('val_mAP_c_cap_ema', mAP_c_cap_ema, epoch)
                # summary_writer.add_scalar('val_mAP_c_ema', mAP_c_ema, epoch)
                
                # summary_writer.add_scalar('val_meanAUC', meanAUC, epoch)
                # summary_writer.add_scalar('val_meanAUC_ema', meanAUC_ema, epoch)
                summary_writer.add_scalar('val_meanAUC_x', meanAUC_x, epoch)
                # summary_writer.add_scalar('val_meanAUC_x_ema', meanAUC_x_ema, epoch)
                summary_writer.add_scalar('val_meanAUC_c_cap', meanAUC_c_cap, epoch)
                # summary_writer.add_scalar('val_meanAUC_c_cap_ema', meanAUC_c_cap_ema, epoch)
                summary_writer.add_scalar('val_meanAUC_c', meanAUC_c, epoch)
                # summary_writer.add_scalar('val_meanAUC_c_ema', meanAUC_c_ema, epoch)

            ## record best meanAUC
            # if meanAUC > best_regular_meanAUC:
            #     best_regular_meanAUC = max(best_regular_meanAUC, meanAUC)
            #     best_regular_epoch = epoch
            # if meanAUC_ema > best_ema_meanAUC:
            #     best_ema_meanAUC = max(meanAUC_ema, best_ema_meanAUC)            
            # if meanAUC_ema > meanAUC:
            #     meanAUC = meanAUC_ema
            #     state_dict = ema_m.module.state_dict()
            # else:
            #     state_dict = model.state_dict()            
            # is_best = meanAUC > best_meanAUC
            # if is_best:
            #     best_epoch = epoch
            # best_meanAUC = max(meanAUC, best_meanAUC)
            # logger.info("{} | Set best meanAUC {} in ep {}".format(epoch, best_meanAUC, best_epoch))
            # logger.info("   | best regular meanAUC {} in ep {}".format(best_regular_meanAUC, best_regular_epoch))
            # if dist.get_rank() == 0:
            #     save_checkpoint({
            #         'epoch': epoch + 1,
            #         # 'arch': args.arch,
            #         'state_dict': state_dict,
            #         # 'best_mAP': best_mAP,
            #         'best_meanAUC': best_meanAUC,
            #         'optimizer': optimizer.state_dict(),
            #     }, is_best=is_best, filename=os.path.join(args.output, 'checkpoint.pth.tar'))
            # # filename=os.path.join(args.output, 'checkpoint_{:04d}.pth.tar'.format(epoch))
            # if math.isnan(loss) or math.isnan(loss_ema):
            #     save_checkpoint({
            #         'epoch': epoch + 1,
            #         # 'arch': args.arch,
            #         'state_dict': model.state_dict(),
            #         # 'best_mAP': best_mAP,
            #         'best_meanAUC': best_meanAUC,
            #         'optimizer': optimizer.state_dict(),
            #     }, is_best=is_best, filename=os.path.join(args.output, 'checkpoint_nan.pth.tar'))
            #     logger.info('Loss is NaN, break')
            #     sys.exit(1)

            ## versione mia TODO: il salvataggio del modello è condizionato alle performance di z_c_cap, ossia la versione con il backdoor adjustement
            if meanAUC_c_cap > best_regular_meanAUC_c_cap:
                best_regular_meanAUC_c_cap = meanAUC_c_cap # this is the best regular, non-EMA
                best_regular_epoch_c_cap = epoch

            # if meanAUC_c_cap_ema > best_ema_meanAUC_c_cap:
            #     best_ema_meanAUC_c_cap = meanAUC_c_cap_ema  

            # if meanAUC_c_cap_ema > meanAUC_c_cap:# If the EMA model is better, then
            #     meanAUC_c_cap = meanAUC_c_cap_ema # overriding
            #     state_dict = ema_m.module.state_dict()
            # else:
            #     state_dict = model.state_dict()  
            state_dict = model.state_dict()  #TODO questa linea va messa solo se commento il else sopra, ovvero solo quando non voglio considerare gli EMA    
                       
            #This is the best globally, EMA or non-EMA:
            is_best = meanAUC_c_cap > best_meanAUC_c_cap
            if is_best:
                best_epoch_c_cap = epoch
            best_meanAUC_c_cap = max(meanAUC_c_cap, best_meanAUC_c_cap)
            logger.info("{} | Set best meanAUC (c_cap) {} in ep {}".format(epoch, best_meanAUC_c_cap, best_epoch_c_cap))
            logger.info("  | best regular (non-EMA) meanAUC (c_cap) {} in ep {}".format(best_regular_meanAUC_c_cap, best_regular_epoch_c_cap))
            if dist.get_rank() == 0:
                save_checkpoint(
                    {
                    'epoch': epoch + 1,
                    'state_dict': state_dict,
                    'best_meanAUC_c_cap': best_meanAUC_c_cap,
                    'optimizer': optimizer.state_dict(),
                    }, is_best=is_best, filename=os.path.join(args.output, 'ckpt_c_cap.pth.tar')
                )

            # if math.isnan(loss) or math.isnan(loss_ema):
            #     save_checkpoint({
            #         'epoch': epoch + 1,
            #         'state_dict': model.state_dict(),
            #         'best_meanAUC_c_cap': best_meanAUC_c_cap,
            #         'optimizer': optimizer.state_dict(),
            #     }, is_best=is_best, filename=os.path.join(args.output, 'ckpt_c_cap_nan.pth.tar'))
            #     logger.info('Loss is NaN, break')
            #     sys.exit(1)

            ## early stop
            # if args.early_stop:
            #     if best_epoch >= 0 and epoch - max(best_epoch, best_regular_epoch) > 8:
            #         if len(ema_mAP_list) > 1 and ema_mAP_list[-1] < best_ema_mAP:
            #             logger.info("epoch - best_epoch = {}, stop!".format(epoch - best_epoch))
            #             if dist.get_rank() == 0 and args.kill_stop:
            #                 filename = sys.argv[0].split(' ')[0].strip()
            #                 killedlist = kill_process(filename, os.getpid())
            #                 logger.info("Kill all process of {}: ".format(filename) + " ".join(killedlist))
            #             break
            
            # versione mia TODO:
            if args.early_stop:
                if best_epoch_c_cap >= 0 and epoch - max(best_epoch_c_cap, best_regular_epoch_c_cap) >= 8:
                    # if len(ema_mAP_c_cap_list) > 1 and ema_mAP_c_cap_list[-1] < best_ema_mAP_c_cap: #TODO ignoro per un attimo tutto ciò che è EMA
                        logger.info("epoch - best_epoch = {}, stop!".format(epoch - best_epoch_c_cap))
                        if dist.get_rank() == 0 and args.kill_stop:
                            filename = sys.argv[0].split(' ')[0].strip()
                            killedlist = kill_process(filename, os.getpid())
                            logger.info("Kill all process of {}: ".format(filename) + " ".join(killedlist))
                        break

    # print("Best mAP_c_cap:", best_mAP_c_cap)

    ### TODO ### added cleaning of process group in DDP
    print("---End of training---")
    # Clean up the process groups:
    dist.destroy_process_group()
    ############

    if summary_writer:
        summary_writer.close()

    return 0


def train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger):

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # batch_time = AverageMeter('T', ':5.3f')
    # data_time = AverageMeter('DT', ':5.3f')
    # speed_gpu = AverageMeter('S1', ':.1f')
    # speed_all = AverageMeter('SA', ':.1f')

    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        len(train_loader),
        # [batch_time, data_time, speed_gpu, speed_all, lr, losses, mem],
        [lr, losses, mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    model.train()

    end = time.time()

    # for i, (images, target) in enumerate(train_loader):
    # for i, (images, target) in enumerate(tqdm(train_loader)):
    for i, (images, target, target_dataset) in enumerate(tqdm(train_loader)): #TODO

        ### TODO Memory leakage debugging
        # print(f"MEM: {torch.cuda.memory_allocated()/(1024*1024)}, MAX: {torch.cuda.max_memory_allocated()/(1024*1024)}")
        # measure data loading time
        # data_time.update(time.time() - end)        


        #TODO 12 Feb moving to cuda device is commented out when using Accelerator
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        target_dataset=target_dataset.cuda(non_blocking=True)

        # if i%3==0:
        #     print(f"train: target in trainloader: {target}")

        # compute output
        with torch.cuda.amp.autocast(enabled=args.amp):
            # print(f"images: {images.size()}")

            ####### VERSIONE LORO, ORIGINALE, COMMENTATA
            # use, half = model(images)            
            #TODO code explain:
            #   'use':  z_x = CLS(h_x), or at least the h_x = MLP_causal(Q)
            #   'half': z_c^= CLS(h_x + h_c^), or at least the the 'h_x + h_c^' (intervened backdoor) 
            # I think we should make the model return also the plain z_c (the actual confoundin features)
            #    indeed, I feel it's wrong to use the current 'half' to perform the KL Loss for variable loss2:
            #       Instead, we would need it to be computed between the uniform and the 'z_c' (not the 'z_c^').           
            # uniform_target = torch.ones_like(half, dtype=torch.float).to('cuda') / args.num_class #we need to push its prediction equally to all categories
            # loss1 = criterion(use, target)
            # loss2 = F.kl_div(F.log_softmax(half, dim=1), uniform_target, reduction='batchmean')
            # loss3 = criterion(half, target)
            # loss = loss1 + args.kl * loss2 + args.ce2 * loss3




            
            #TODO 13 ottobre, MIA VERSIONE CHE TENTA DI CORREGGERE LA LORO IN ACCORDO CON QUANTO DICHIARATO SUL PAPER:
            z_x, z_c_cap, z_c = model(images)
            



            ## The supervised loss regards causal features
            loss1 = criterion(z_x, target) #the supervised loss
            if torch.isnan(loss1).any():
                logger.info("TRAIN Loss - loss1 has NaNs: raise ValueError and exit")
                raise ValueError

            ## The confounding features get associated with a uniform probability over all classes
            z_c_log_sm = F.log_softmax(z_c, dim=1)
            # uniform_target = torch.ones_like(z_c_log_sm, dtype=torch.float).to('cuda') / args.num_class #we need to push its prediction equally to all categories
            uniform_target = torch.ones_like(z_c_log_sm, dtype=torch.float, device='cuda') / args.num_class #
            loss2 = F.kl_div(z_c_log_sm, uniform_target, reduction='batchmean')
            if torch.isnan(loss2).any():
                logger.info("TRAIN Loss - loss2 has NaNs: raise ValueError and exit")
                raise ValueError
            
            ## The causally-intervened features are forced to align with the actual target to make them robust to confounding features
            loss3 = criterion(z_c_cap, target)
            # Embed all the terms together
            # print(f"    z_x: {z_x}\nz_c_cap: {z_c_cap}\n    z_c: {z_c}\nz_c_logsm: {z_c_log_sm}")
            if torch.isnan(loss3).any():
                logger.info("TRAIN Loss - loss3 has NaNs: raise ValueError and exit")
                raise ValueError
            
            loss = loss1 + args.kl*loss2 + args.ce2*loss3
            ####fine mia versione


            #TODO commented out this printout
            # if random.random()<=0.1:
            #     # logger.info('loss1: {:.3f} loss2: {:.3f} loss3: {:.3f}'.format(loss1, loss2, loss3)) #TODO
            #     logger.info(f'gpu-rank: {args.local_rank}| loss1: {loss1} loss2: {loss2} loss3: {loss3}')

            if args.loss_dev > 0:
                loss *= args.loss_dev

        # record loss
        # losses.update(loss.item(), images.size(0))
        losses.update(loss, images.size(0)) #TODO

        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # compute gradient and do SGD step
        # optimizer.zero_grad()
        optimizer.zero_grad(set_to_none=True) #TODO 20 oct

        # del images, target
        # torch.cuda.empty_cache()

        scaler.scale(loss).backward() #TODO 12 Feb replace with accelerator
        # accelerator.backward(scaler.scale(loss))

        scaler.step(optimizer)
        scaler.update()
        
        del images, target, loss, loss1, loss2, loss3 #TODO
        torch.cuda.empty_cache()


        
        # # one cycle learning rate #TODO loro lo mettevano dentro il FOR dei batch... invece io voglio aggiornare LR solo dopo l'epoca
        scheduler.step()
        
        
        





        lr.update(get_learning_rate(optimizer))
        # if epoch >= args.ema_epoch: #TODO ignore EMA just for now
        #     ema_m.update(model)

        ## measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()
        # speed_gpu.update(images.size(0) / batch_time.val, batch_time.val)
        # speed_all.update(images.size(0) * dist.get_world_size() / batch_time.val, batch_time.val)

        if i % args.print_freq == 0:
            progress.display(i, logger)

    # #TODO replaced one cycle learning rate with the following to see the old and the new value of LR as it proceeds during epochs
    #     if epoch in [0, round(0.25*args.epochs), round(0.50*args.epochs), round(0.75*args.epochs)]:
    #         before_lr = optimizer.param_groups[0]["lr"]
    #         scheduler.step()
    #         after_lr = optimizer.param_groups[0]["lr"]
    #         print(f"Epoch {epoch}: lr {before_lr}--->{after_lr}")
    #     else:#update anyway but does not show any message
    #         scheduler.step()
    return losses.avg


@torch.no_grad()
def validate(val_loader, model, criterion, args, logger, epoch):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, mem],
        prefix='Test: ')

    # switch to evaluate mode
    saveflag = False
    model.eval()

    # saved_data = []
    saved_data_x = []
    saved_data_c_cap = []
    saved_data_c = []

    targets = None

    # outputs = None
    outputs_x = None
    outputs_c_cap = None
    outputs_c = None

    with torch.no_grad():
        end = time.time()
        # for i, (images, target) in enumerate(val_loader):
        for i, (images, target, target_dataset) in enumerate(val_loader): #TODO
            
            #TODO 12 Feb 2024: moving to cuda device is commented out when using Accelerator:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            target_dataset=target_dataset.cuda(non_blocking=True)
            
            # if i%3==0:
            #     print(f"valid: target in validloader: {target}")

            # compute output
            with torch.cuda.amp.autocast(enabled=args.amp):
                ##### VERSIONE LORO ORIGINALE COMMENTATA
                # use, half = model(images)
                # uniform_target = torch.ones_like(half, dtype=torch.float).to('cuda') / args.num_class
                # loss1 = criterion(use, target)
                # loss2 = F.kl_div(torch.log_softmax(half, dim=1), uniform_target, reduction='batchmean')
                # loss3 = criterion(half, target)
                # loss = loss1 + args.kl * loss2 + args.ce2 * loss3
                # #TODO 13 ottobre, MIA VERSIONE CHE TENTA DI CORREGGERE LA LORO IN ACCORDO CON QUANTO DICHIARATO SUL PAPER:
                # z_x, z_c_cap, z_c = model(images)
                # uniform_target = torch.ones_like(z_c, dtype=torch.float).to('cuda') / args.num_class #we need to push its prediction equally to all categories
                # loss1 = criterion(z_x, target) #the supervised loss
                # loss2 = F.kl_div(F.log_softmax(z_c, dim=1), uniform_target, reduction='batchmean')
                # loss3 = criterion(z_c_cap, target)
                # loss = loss1 + args.kl * loss2 + args.ce2 * loss3
                



                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof_validate:
                #     with record_function("VALIDATE_model"):
                # #TODO 17 ottobre, altra MIA VERSIONE:
                #         z_x, z_c_cap, z_c = model(images)
                # print(prof_validate.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=15))
                
                z_x, z_c_cap, z_c = model(images)
                
                loss1 = criterion(z_x, target) #the supervised loss
                if torch.isnan(loss1).any():
                    
                    logger.info("VALID Loss - loss1 has NaNs: raise ValueError and exit")
                    raise ValueError
                
                # uniform_target = torch.ones_like(z_c, dtype=torch.float).to('cuda') / args.num_class #we need to push its prediction equally to all categories
                uniform_target = torch.ones_like(z_c, dtype=torch.float, device='cuda') / args.num_class #we need to push its prediction equally to all categories
                loss2 = F.kl_div(F.log_softmax(z_c, dim=1), uniform_target, reduction='batchmean')
                if torch.isnan(loss2).any():
                    logger.info("VALID Loss - loss2 has NaNs: raise ValueError and exit")
                    raise ValueError
                
                loss3 = criterion(z_c_cap, target)
                if torch.isnan(loss3).any():
                    logger.info("VALID Loss - loss3 has NaNs: raise ValueError and exit")
                    raise ValueError
                
                loss = loss1 + args.kl * loss2 + args.ce2 * loss3

                # TODO commented out this printout
                # if random.random()<=0.1:
                #     # logger.info('testloss1: {:.3f} loss2: {:.3f} loss3: {:.3f}'.format(loss1, loss2, loss3)) #TODO
                #     logger.info(f'(test) gpu-rank: {args.local_rank}| loss1: {loss1} loss2: {loss2} loss3: {loss3}')

                if args.loss_dev > 0:
                    loss *= args.loss_dev

                # output_sm = torch.sigmoid(use)######## TODO un mio commento: quindi palesemente prende solo le probabilità dell'output causale z_x
                ## ma ci servono anche le altre due se voglio fare un grafico come quello nel paper
                #TODO 13 ottobre, MIA VERSIONE:
                output_sm_x = torch.sigmoid(z_x)
                output_sm_c_cap = torch.sigmoid(z_c_cap)
                output_sm_c = torch.sigmoid(z_c)
                
                
                if torch.isnan(loss):
                    saveflag = True
            tar = target.cpu()

            # out = output_sm.cpu()
            out_x = output_sm_x.cpu()
            out_c_cap = output_sm_c_cap.cpu()
            out_c = output_sm_c.cpu()
            # print(f"VALIDATE - out_x: {out_x}, out_c_cap: {out_c_cap}, out_c: {out_c}")

            targets = tar if targets == None else torch.cat([targets, tar])
            # outputs = out if outputs == None else torch.cat([outputs, out])
            outputs_x = out_x if outputs_x == None else torch.cat([outputs_x, out_x])
            outputs_c_cap = out_c_cap if outputs_c_cap == None else torch.cat([outputs_c_cap, out_c_cap])
            outputs_c = out_c if outputs_c == None else torch.cat([outputs_c, out_c])

            # record loss
            # losses.update(loss.item() * args.batch_size, images.size(0))
            losses.update(loss* args.batch_size, images.size(0)) #TODO

            mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

            # # save some data
            # # output_sm = nn.functional.sigmoid(output)
            # # _item = torch.cat((output_sm.detach().cpu(), target.detach().cpu()), 1)            
            # _item_x = torch.cat((output_sm_x.detach().cpu(), target.detach().cpu()), 1)
            # _item_c_cap = torch.cat((output_sm_c_cap.detach().cpu(), target.detach().cpu()), 1)
            # _item_c = torch.cat((output_sm_c.detach().cpu(), target.detach().cpu()), 1)
            _item_x = torch.cat((out_x, tar), 1)
            _item_c_cap = torch.cat((out_c_cap, tar), 1)
            _item_c = torch.cat((out_c, tar), 1)


            #del output_sm
            #del target
            # saved_data.append(_item)
            saved_data_x.append(_item_x)
            saved_data_c_cap.append(_item_c_cap)
            saved_data_c.append(_item_c)


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and dist.get_rank() == 0:
                progress.display(i, logger)

        logger.info('=> synchronize...')
        if dist.get_world_size() > 1:
            dist.barrier()
        loss_avg, = map(
            _meter_reduce if dist.get_world_size() > 1 else lambda x: x.avg,
            [losses]
        )

        # import ipdb; ipdb.set_trace()
        ## calculate mAP
        # saved_data = torch.cat(saved_data, 0).numpy()
        saved_data_x = torch.cat(saved_data_x, 0).numpy()
        saved_data_c_cap = torch.cat(saved_data_c_cap, 0).numpy()
        saved_data_c = torch.cat(saved_data_c, 0).numpy()

        # saved_name = 'saved_data_tmp.{}.txt'.format(dist.get_rank())
        saved_name_x = 'saved_data_x_{}.txt'.format(dist.get_rank())
        saved_name_c_cap = 'saved_data_c_cap_{}.txt'.format(dist.get_rank())
        saved_name_c = 'saved_data_c_{}.txt'.format(dist.get_rank())

        # np.savetxt(os.path.join(args.output, saved_name), saved_data) #original
        np.savetxt(os.path.join(args.output, saved_name_x), saved_data_x)
        np.savetxt(os.path.join(args.output, saved_name_c_cap), saved_data_c_cap)
        np.savetxt(os.path.join(args.output, saved_name_c), saved_data_c)



        if dist.get_world_size() > 1:
            dist.barrier()

        if dist.get_rank() == 0:
            print("Calculating mAP:")
            ## filenamelist = ['saved_data_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            filenamelist_x = ['saved_data_x_{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            filenamelist_c_cap = ['saved_data_c_cap_{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            filenamelist_c = ['saved_data_c_{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            
            
            metric_func = voc_mAP

            ## mAP, aps = metric_func([os.path.join(args.output, _filename) for _filename in filenamelist], args.num_class,return_each=True) #original
            mAP_x, aps_x = metric_func([os.path.join(args.output, _filename) for _filename in filenamelist_x], args.num_class,return_each=True)
            mAP_c_cap, aps_c_cap = metric_func([os.path.join(args.output, _filename) for _filename in filenamelist_c_cap], args.num_class,return_each=True)
            mAP_c, aps_c = metric_func([os.path.join(args.output, _filename) for _filename in filenamelist_c], args.num_class,return_each=True)
            

            # logger.info("  mAP: {}".format(mAP))
            # logger.info("   aps: {}".format(np.array2string(aps, precision=5)))
            logger.info("  meanAP (x): {}".format(np.array2string(mAP_x, precision=3)))
            ## logger.info("   aps (x): {}".format(np.array2string(aps_x, precision=4)))
            logger.info("  meanAP (c_cap): {}".format(np.array2string(mAP_c_cap, precision=3)))
            ## logger.info("   aps (c_cap): {}".format(np.array2string(aps_c_cap, precision=4)))
            logger.info("  meanAP (c): {}".format(np.array2string(mAP_c,precision=3)))
            # logger.info("   aps (c): {}".format(np.array2string(aps_c, precision=4)))
        else:
            # mAP = 0
            mAP_x=0
            mAP_c_cap=0
            mAP_c=0

        if dist.get_world_size() > 1:
            dist.barrier()
    # outputs = outputs.detach().numpy()
    outputs_x = outputs_x.detach().numpy()
    outputs_c_cap = outputs_c_cap.detach().numpy()
    outputs_c = outputs_c.detach().numpy()

    targets = targets.detach().numpy()

    # print(targets) #TODO debugging

    # auc_scores = roc_auc_score(targets, outputs, average=None) #TODO code explanation: get the AUROC score for each class
    # mean = (sum(auc_scores)-auc_scores[12]) / 14 #TODO code explanation: subtract the score of the NoFind class (12th position) and divide by (n.classes-1)=14
    # meanAUC = mean
    auc_scores_x = roc_auc_score(targets, outputs_x, average=None) #TODO code explanation: get the AUROC score for each class
    auc_scores_c_cap = roc_auc_score(targets, outputs_c_cap, average=None)
    auc_scores_c = roc_auc_score(targets, outputs_c, average=None)
    NOFINDING_position_in_list=cate.index('No Finding')
    meanAUC_x = (sum(auc_scores_x)-auc_scores_x[NOFINDING_position_in_list]) / (args.num_class-1) #TODO May 2024
    meanAUC_c_cap = (sum(auc_scores_c_cap)-auc_scores_c_cap[NOFINDING_position_in_list]) / (args.num_class-1)
    meanAUC_c = (sum(auc_scores_c)-auc_scores_c[NOFINDING_position_in_list]) / (args.num_class-1)

    # logger.info("AUC {}".format(auc_scores))
    #logger.info("meanAUC {}".format(meanAUC))
    # logger.info("AUROC scores: {}".format(auc_scores))
    # logger.info("mean AUROC {}".format(meanAUC))

    # logger.info("AUROC scores (x): {}".format(auc_scores_x))
    logger.info("mean AUROC (x): {}".format(meanAUC_x))
    # logger.info("AUROC scores (c_cap): {}".format(auc_scores_c_cap))
    logger.info("mean AUROC (c_cap): {}".format(meanAUC_c_cap))
    # logger.info("AUROC scores (c): {}".format(auc_scores_c))
    logger.info("mean AUROC (c): {}".format(meanAUC_c))
    

    #Already commented by the original authors:
    #  cates = ["Car", "Nod", "Fib", "Pna", "Her", "Ate", "Pnx", "Inf",
    #          "Mas", "Ple", "Ede", "Con", "Nofind", "Emp", "Eff"]  # 416:12
    # # cates = ['Car', 'Pnx', 'Con', 'Mas', 'Ple', 'Inf', 'Ede',
    # #         'Her', 'Fib', 'NoF', 'Emp', 'Pna', 'Nod', 'Ate', 'Eff']#256:9
    # for idx, cate in enumerate(cates):
    #     # epoch = "test"
    #     if idx == 12:
    #         continue
    #     yp = outputs[:, idx]
    #     yl = targets[:, idx]
    #     y_label = np.array(yl)
    #     y_pred = np.array(yp)
    #     fpr = dict()
    #     tpr = dict()
    #     roc_auc = dict()
    #     fpr[0], tpr[0], _ = roc_curve(y_label, y_pred)
    #     roc_auc[0] = auc(fpr[0], tpr[0])
    #     lw = 2
    #     method_name = cates[idx]
    #     plt.figure(epoch)
    #     plt.plot(fpr[0], tpr[0],
    #              lw=lw, label=method_name + ' (%0.3f)' % roc_auc[0])
    #     # plt.plot(fpr[0], tpr[0],
    #     #          lw=lw, label=method_name)
    #     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     fontsize = 14
    #     plt.xlabel('False Positive Rate', fontsize=fontsize)
    #     plt.ylabel('True Positive Rate', fontsize=fontsize)
    #     plt.legend(loc=0, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
    #     plt.tight_layout()
    #     figure_file = "auc"+"%0.6f"%mean+"-epoch{}".format(epoch)
    #     dir = args.output + "/plt/"
    #     if not os.path.exists(dir): os.makedirs(dir)
    #     plt.savefig(os.path.join(dir, "%s.png" % figure_file))
    
    # return loss_avg, mAP, meanAUC
    ##Versione mia TODO:
    return loss_avg, mAP_x, meanAUC_x, mAP_c_cap, meanAUC_c_cap, mAP_c, meanAUC_c
    # ##TODO:
    # return loss_avg


##################################################################################
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

##Just for now, do not utilize EMA models:
# class ModelEma(torch.nn.Module):
#     def __init__(self, model, decay=0.9997, device=None):
#         super(ModelEma, self).__init__()
#         # make a copy of the model for accumulating moving average of weights
#         self.module = deepcopy(model)
#         self.module.eval()

#         # import ipdb; ipdb.set_trace()

#         self.decay = decay
#         self.device = device  # perform ema on different device from model if set
#         if self.device is not None:
#             self.module.to(device=device)

#     def _update(self, model, update_fn):
#         with torch.no_grad():
#             for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
#                 if self.device is not None:
#                     model_v = model_v.to(device=self.device)
#                 ema_v.copy_(update_fn(ema_v, model_v))

#     def update(self, model):
#         self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

#     def set(self, model):
#         self._update(model, update_fn=lambda e, m: m)


def _meter_reduce(meter):
    #Original
    # meter_sum = torch.FloatTensor([meter.sum]).cuda()
    # meter_count = torch.FloatTensor([meter.count]).cuda()
    # My first
    # meter_sum = torch.FloatTensor([meter.sum],device=torch.device('cuda'))
    # meter_count = torch.FloatTensor([meter.count], device=torch.device('cuda'))
    #But then:
    meter_sum = torch.Tensor([meter.sum]).cuda()
    meter_count = torch.Tensor([meter.count]).cuda()

    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.split(filename)[0] + '/model_best.pth.tar')
        # shutil.copyfile(filename, os.path.split(filename)[0] + '/model_best.pth.tar')


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
    #TODO
    del lines
    #
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


if __name__ == '__main__':
    print("Sto iniziando\n")   
     
    args = get_args()

    # world_size=args.number_of_gpus
    # mp.spawn(main, args=(world_size,), nprocs=world_size) #TODO May 2024: if you use sinle(mono)-node multi-GPU it is ok to use multiprocessing.spawn
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    main(args.rank, args.world_size, args) # but when you wish to leverage multi-node multi-gpu clusters, you just call del main (DDP will do)