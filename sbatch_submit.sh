#!/bin/bash

NUM_NODES=3 #3
NUM_GPUS_PER_NODE=4 #4

sbatch -A IscrC_CAFE \
    --job-name caumedGC \
    -N $NUM_NODES \
    --ntasks-per-node=$((8*$NUM_GPUS_PER_NODE)) \
    --gpus-per-node=$NUM_GPUS_PER_NODE \
    --mem=463000 \
    -p boost_usr_prod \
    -t 1440 \
    -o $PWD/out.out \
    -e $PWD/err.err \
    $PWD/run_train.sh $NUM_GPUS_PER_NODE
#--gpus-per-node=$NUM_GPUS_PER_NODE \