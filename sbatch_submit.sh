#!/bin/bash

############ Multi-node multi-GPU job
#Got inspiration from:
#   -  https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904
#   -  http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-torch-multi-eng.html

#### For a reservation of N four-GPU A100 nodes:
NNODES=4 #4 #4,3, or 1 #total number of nodes
NGPUSPERNODE=4 #This directive specifies the number of GPUs you want to reserve per node. In your case, you have 4 GPUs per node, so requesting 4 allows your DDP training process to utilize all available GPUs on the allocated node.

#### It is important to understand also the CPU cores and tasks.
#If your training job is GPU-bound and each task does not require intensive CPU computation --> go for Option 2
#If your training job involves significant CPU processing or data loading that benefits from more CPU cores --> go for Option 1
#Please, make your choice below and comment the other:
# Would you prefer Option 1 (--ntasks-per-node=1, --cpus-per-task=32) ???:
NTASKSPERNODE=1 # number of tasks per node (recommended is 1 for DDP)
NCPUSPERTASK=32 #While each DDP process manages a single GPU, it can still benefit from multiple CPU cores for data loading, preprocessing, and other operations alongside training.
# or Option 2 (--ntasks-per-node=4, --cpus-per-task=8) ???:
#
# NTASKSPERNODE=4 # if you use 4 tasks it means or 1 task per GPU
# NCPUSPERTASK=8
####

sbatch -A IscrC_CAFE \
    --job-name=caumedGC \
    --nodes=$NNODES \
    --ntasks-per-node=$NTASKSPERNODE \
    --cpus-per-task=$NCPUSPERTASK \
    --gpus-per-node=$NGPUSPERNODE \
    --mem=124000 \
    -p boost_usr_prod \
    -t 1440 \
    -o $PWD/out.out \
    -e $PWD/err.err \
    $PWD/run_train.sh $NGPUSPERNODE
#--gpus-per-node=$NGPUSPERNODE \
# --mem=490000 \
