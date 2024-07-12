#!/bin/bash
####### Please, DO NOT uncomment the following lines beginning with #SBATCH. Indeed, they are not comments, but rather interpreted as commands (#+SBATCH)
#SBATCH -A YourHPCAccount #(optional)
#SBATCH --job-name=YourExperimentName
#SBATCH -p YourHPCPartition

#SBATCH --nodes=6 #Number of computing nodes, e.g., six

### The product of the two following terms must be <=32 (which is the number of CPU cores on each computing node of our cluster: please, adapt if needed)
#SBATCH --cpus-per-task=8 #Number of CPU cores per task, e.g., 8, 16, 32
#SBATCH --ntasks-per-node=4 #Number of tasks/threads per node, e.g., 4, 2, 1

#SBATCH --gpus-per-node=4 ### This must be set equal to --ntasks-per-node to make each process/thread/task handle a separate GPU, thus being more efficient with PyTorch's DDP
#SBATCH --time=1440 #Minutes (Optional), its maximum values is set to the partition's time limit, see you Slurm configuration
#SBATCH --mem=0 #Megabytes, e.g., 490000. If set =0, it means all the available memory of the node(s)
#SBATCH --output=out_file.out # Optional, redirect the output of the script to this file
#SBATCH --error=err_file.err # Optional, redirect the errors of the script to this file

# Attention! When using DDP with more than one computing node and/or more than one GPU, you need to set this batch size variable appropriately.
# For instance, if you want to obtain a batch size of 12 on each device, you need to assign this variable to 12*number_of_nodes*gpus_per_node
# E.g., using one node with four GPUs would be = 12*(1*4)=48;
# E.g., using six nodes with four GPUs would be = 12*(6*4)=288;
# E.g., using 24 nodes with one GPU each, would be = 12*(24*1)=288;
BATCH_SIZE=288

### Now, we need to export some important variables to set torch DDP appropriately
# In the following, change the 5-digit MASTER_PORT value as you wish, SLURM will raise Error if duplicated with others
# And assign WORLD_SIZE as =number_of_nodes*gpus_per_node, e.g., 6*4=24
export MASTER_PORT=12340
export WORLD_SIZE=24 #### 24, 1

### Get the first node name as the master node address
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### The command to run the main CROCODILE python training script.
# Please, adapt the path of your python.exe depending on your conda environment. 

srun ~/.conda/envs/caumed/bin/python main-crocodile.py \
                                --dataset_dir '../' \
                                --output jun20ID \
                                --backbone resnet50 \
                                --adjustContrast \
                                --optim AdamW \
                                --num_class 9 \
                                --img_size 320 \
                                --batch-size $BATCH_SIZE \
                                --subset_take1everyN 1 \
                                --print-freq 100 \
                                --seed 42 \
                                --epochs 30 \
                                --workers 8 \
                                --weight-decay 0.00015 \
                                --lr 3e-6 \
                                --dropoutrate_randomaddition 0.3 \
                                --early-stop \
                                --pretrained \
                                --useCrocodile \
                                --num_class_crocodile 3 \
                                --useContrastiveLoss \
                                --crocodile_CODI_common_dim 16 \
                                --contrastiveLossSpace 'representation' \
                                --useCausalityMaps \
                                --w_CE_y 1.0 \
                                --w_KL_y 10 \
                                --w_CE_y_bd 1.0 \
                                --w_CE_d 1.0 \
                                --w_KL_d 86 \
                                --w_CE_d_bd 1.0 \
                                --w_Y_imgLvl 110 \
                                --w_Yca_sameY 25.0 \
                                --w_Ysp_diffY 15.0 \
                                --w_Dca_sameD 25.0 \
                                --w_Dsp_diffD 15.0 \
                                --w_Yprior 1000