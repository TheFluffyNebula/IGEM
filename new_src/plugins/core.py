import torch
import time
from typing import Callable
def time_projection(func: Callable, **kwargs):
    torch.cuda.synchronize()                   # 1) wait for any pending kernels
    t0 = time.perf_counter()                   # 2) stamp start
    g_proj = func(**kwargs)
    torch.cuda.synchronize()                   # 3) wait for finish
    t1 = time.perf_counter()                   # 4) stamp end
    return g_proj, (t1 - t0)
