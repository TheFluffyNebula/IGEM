from typing import Dict, Any
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from avalanche.models import avalanche_forward
import qpsolvers
from .base_gem import BaseGEMPlugin
from . import core

class GEMPlugin(BaseGEMPlugin):
    def __init__(
        self,
        patterns_per_exp: int,
        memory_strength: float,
        proj_interval: int,
        proj_metric: Any,
    ):
        print("=======Using GEM plugin======")
        super().__init__(
            memory_strength=memory_strength,
            proj_interval=proj_interval,
            patterns_per_exp=patterns_per_exp,
            proj_metric=proj_metric,
        )
        self.memory_x = {}
        self.memory_y = {}
        self.memory_tid = {}

    def _has_memory(self) -> bool:
        return bool(self.memory_x)
    
    @torch.no_grad()
    def _update_memory(self, strategy):
        dataset = strategy.experience.dataset
        t = strategy.clock.train_exp_counter
        batch_size = strategy.train_mb_size
        
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
        )
        tot = 0
        for mbatch in dataloader:
            x, y, tid = mbatch[0], mbatch[1], mbatch[-1]
            if tot + x.size(0) <= self.patterns_per_exp:
                if t not in self.memory_x:
                    self.memory_x[t] = x.clone()
                    self.memory_y[t] = y.clone()
                    self.memory_tid[t] = tid.clone()
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x), dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], y), dim=0)
                    self.memory_tid[t] = torch.cat((self.memory_tid[t], tid), dim=0)

            else:
                diff = self.patterns_per_exp - tot
                if t not in self.memory_x:
                    self.memory_x[t] = x[:diff].clone()
                    self.memory_y[t] = y[:diff].clone()
                    self.memory_tid[t] = tid[:diff].clone()
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x[:diff]), dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], y[:diff]), dim=0)
                    self.memory_tid[t] = torch.cat(
                        (self.memory_tid[t], tid[:diff]), dim=0
                    )
                break
            tot += x.size(0)

    def _should_project(self, g, mem_strength):
        return (torch.mv(self.reference, g) < -mem_strength).any()
    
    def _solve_projection(self, g: Tensor, reference: Tensor, memory_strength: float):
        #print("Solving projection...")
        # reference: G matrix
        memories_np = reference.cpu().double().numpy()
        gradient_np = g.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + memory_strength
        v_star = qpsolvers.solve_qp(P=P, q=-q, G=-G.transpose(), h=-h, solver="quadprog")
        g_proj = np.dot(v_star, memories_np) + gradient_np
        return torch.from_numpy(g_proj).float().to(g.device)
    
    def _compute_reference_gradients(self, strategy) -> Tensor:
        grads = []
        for t in range(strategy.clock.train_exp_counter):
            strategy.optimizer.zero_grad()
            xref = self.memory_x[t].to(strategy.device)
            yref = self.memory_y[t].to(strategy.device)
            out = avalanche_forward(strategy.model, xref, self.memory_tid[t])
            loss = strategy._criterion(out, yref)
            loss.backward()

            flat = [
                (p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=strategy.device))
                for p in strategy.model.parameters()
            ]
            grads.append(torch.cat(flat, dim=0))
        return torch.stack(grads, dim=0)

