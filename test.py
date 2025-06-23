import torch

if hasattr(torch.cuda.nccl, "version"):
    print("torch.cuda.nccl.version():", torch.cuda.nccl.version())
