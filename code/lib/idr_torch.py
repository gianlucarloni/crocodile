#!/usr/bin/env python
# coding: utf-8
    
import os
# import hostlist
    
# get SLURM variables
rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])
size = int(os.environ['SLURM_NTASKS'])
cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
    
# # get node list from slurm: 
# hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
    
# # get IDs of reserved GPU: utilize either SLURM variables 'SLURM_JOB_GPUS' or 'SLURM_STEP_GPUS' depending on you SLURM version and system setting
# gpu_ids = os.environ['SLURM_JOB_GPUS'].split(",")
    
# # define MASTER_ADD & MASTER_PORT
# os.environ['MASTER_ADDR'] = hostnames[0]
# os.environ['MASTER_PORT'] = str(12345 + int(min(gpu_ids))) # to avoid port conflict on the same node