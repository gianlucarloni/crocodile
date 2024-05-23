#!/bin/bash

salloc --job-name=crocoSalloc \
	-p boost_usr_prod \
	--nodes=4 \
	--cpus-per-task=32 \
	--ntasks-per-node=1 \
	--gpus-per-node=1 \
	--time=20
	# --mem=10000

srun --jobid=$SLURM_JOBID --pty /bin/bash 

export MASTER_PORT=12340
export BATCH_SIZE=4 # I set this batch size as 8 * number_of_nodes * gpus_per_node. E.g. = 8*(3*4)=96; 24:192, 64:512
export WORLD_SIZE=4 ### change WORLD_SIZE as gpus/node * num_nodes
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### the command to run
srun ~/.conda/envs/caumed/bin/python main-crocodile.py \
                                --dataset_dir '../' \
                                --backbone resnet50 \
                                --optim AdamW \
                                --num_class 9 \
                                --img_size 384 \
                                --batch-size $BATCH_SIZE \
                                --subset_take1everyN 1 \
                                --print-freq 100 \
                                --seed 42 \
                                --epochs 12 \
                                --weight-decay 0.0001 \
                                --lr 1e-5 \
                                --dropoutrate_randomaddition 0.3 \
                                --early-stop \
                                --pretrained \
                                --useCrocodile \
                                --num_class_crocodile 3 \
                                --useContrastiveLoss \
                                --crocodile_CODI_common_dim 16 \
                                --contrastiveLossSpace 'representation'