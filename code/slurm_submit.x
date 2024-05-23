#!/bin/bash
#SBATCH -A IscrC_CAFE
#SBATCH --job-name=crocoMNMG
#SBATCH -p boost_usr_prod

#SBATCH --nodes=6 #$NNODES    # Number of nodes

### The product of the two following terms must be <=32 (which is the number of CPU cores on each cluster computing node)
#SBATCH --cpus-per-task=8 #$NCPUSPERTASK  # number of cores per task
#SBATCH --ntasks-per-node=4 #$NTASKSPERNODE # Number of tasks per node
#SBATCH --gpus-per-node=4 ### This must be set equal to --ntasks-per-node to make each process handle a separate GPU and thus be more efficient with DDP

#SBATCH --time=1440 #minutes

#SBATCH --mem=0 #490000

#SBATCH --output=out-crocoMNMG.out
#SBATCH --error=err-crocoMNMG.err


# I set this batch size as 8 * number_of_nodes * gpus_per_node. E.g. = 8*(3*4)=96; 24:192, 64:512
BATCH_SIZE=192

###OldmySBATCH --ntasks=2 #$NTASKS  # Total number of tasks (world_size) such as 3 nodes of 4 GPU each equals 12
###OldmySBATCH --gres=gpu:2 #$NGPUSPERNODE  # Request n GPUs per node (gres specifies the number of GPUs on a single-node, even in the multi-node case)



### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=24

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


# Only for debugging: 
# export NCCL_DEBUG=INFO
# export CUDA_LAUNCH_BLOCKING=1

### the command to run
# srun ~/.conda/envs/caumed/bin/python main-crocodile.py $NUM_GPUS \
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