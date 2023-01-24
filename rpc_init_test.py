import torch
import torch.distributed as dist
from torch.distributed import rpc 
import os

# Script to initialise an RPC connection for multiple nodes and then print how many GPUs are on each node
# When running this job through slurm, slurm should initialise a "WORLD_SIZE" environment variable based on 
# --nodes and --tasks-per-node where WORLD_SIZE = --nodes x --tasks-per-node
# if --tasks-per-node is not included it will assume that it is 1 task per node and so WORLD_SIZE = --nodes (arg)

# Initialize the RPC group
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
rpc.init_rpc(name=None, rank=rank, world_size=world_size, rpc_backend_options=None)

# Print the number of GPUs on each node
for i in range(world_size):
    if rank == i:
        print(f"Node {i} has {torch.cuda.device_count()} GPUs")
