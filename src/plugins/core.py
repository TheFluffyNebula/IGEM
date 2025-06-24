import numpy as np
import qpsolvers
import torch
import time
from typing import Callable

def timed_solve(func : Callable, **func_kwargs):
    torch.cuda.synchronize()                  
    t0 = time.perf_counter()                  
    result = func(**func_kwargs)
    torch.cuda.synchronize()                  
    t1 = time.perf_counter()                  
    return (result, t1 - t0)
    
def solve_quadprog(self, g):
    """
    Solve quadratic programming with current gradient g and
    gradients matrix on previous tasks G.
    Taken from original code:
    https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py
    """
    memories_np = self.G.cpu().double().numpy()
    gradient_np = g.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + self.memory_strength
    v = qpsolvers.solve_qp(P=P, q=-q, G=-G.transpose(), h=-h, solver="quadprog")
    v_star = np.dot(v, memories_np) + gradient_np
    return torch.from_numpy(v_star).float()


