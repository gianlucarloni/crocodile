#!/bin/bash

NUM_GPUS=$(($1)) #the first (and only) input keyboard argument that is passed to this .sh file
BATCH_SIZE=$((16*$NUM_GPUS))

## TRAIN ## utilizes the conda environment called "genv" (gianluca env)
# ~/.conda/envs/genv/bin/python main-causal_gc.py $NUM_GPUS \
# ~/.conda/envs/caumed/bin/python main-crocodile.py $NUM_GPUS \
# ~/.conda/envs/caumed/bin/python main-debug.py $NUM_GPUS \
## Select from the lines above the one you want to run
~/.conda/envs/caumed/bin/python main-crocodile.py $NUM_GPUS \
                                --world_size $SLURM_NTASKS \
                                --rank $SLURM_PROCID \
                                --dataset_dir './' \
                                --backbone resnet50 \
                                --optim AdamW \
                                --num_class 9 \
                                --img_size 448 \
                                --batch-size $BATCH_SIZE \
                                --workers 8 \
                                --subset_take1everyN 1 \
                                --print-freq 100 \
                                --seed 42 \
                                --epochs 30 \
                                --weight-decay 0.0001618 \
                                --lr 8e-6 \
                                --dropoutrate_randomaddition 0.3 \
                                --early-stop \
                                --pretrained \
                                --useCrocodile