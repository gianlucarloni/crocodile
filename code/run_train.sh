#!/bin/bash

NUM_GPUS=$(($1)) #the first (and only) input keyboard argument that is passed to this .sh file
BATCH_SIZE=$((8*$NUM_GPUS))


# Set the address of the master node (typically the first node)
# Set a port number for communication between nodes
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}
# echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"

## TRAIN ## utilizes the conda environment called "genv" (gianluca env)
# ~/.conda/envs/genv/bin/python main-causal_gc.py $NUM_GPUS \
# ~/.conda/envs/caumed/bin/python main-crocodile.py $NUM_GPUS \
# ~/.conda/envs/caumed/bin/python main-debug.py $NUM_GPUS \
## Select from the lines above the one you want to run

                                # --world_size $SLURM_NTASKS \
                                # --rank $SLURM_PROCID \
~/.conda/envs/caumed/bin/python main-crocodile.py $NUM_GPUS \
                                --dataset_dir '../' \
                                --backbone resnet50 \
                                --optim AdamW \
                                --num_class 9 \
                                --img_size 64 \
                                --batch-size $BATCH_SIZE \
                                --workers 1 \
                                --subset_take1everyN 1 \
                                --print-freq 100 \
                                --seed 42 \
                                --epochs 2 \
                                --weight-decay 0.0001618 \
                                --lr 8e-6 \
                                --dropoutrate_randomaddition 0.3 \
                                --early-stop \
                                --pretrained \
                                --useCrocodile \
				--num_class_crocodile 3 \
				--useContrastiveLoss \
				--crocodile_CODI_common_dim 16 \
# > python_out.log 2> python_err.log
# Check if the Python script ran successfully
if [ $? -ne 0 ]; then
	echo "Python script failed to execute."
fi

### --dataset_dir './' \