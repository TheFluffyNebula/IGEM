from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from .base_gem import BaseGEMPlugin
# import core  

class IGEMPlugin(BaseGEMPlugin):
    def __init__(
        self,
        patterns_per_exp: int,
        pgd_iterations: int,
        lr: float,
        use_adaptive_lr: bool,
        use_warm_start: bool,
        memory_strength: float,
        proj_interval: int,
        proj_metric: Any,
        n_experiences: int,
        memory_size: int,
    ):
        super().__init__(
            n_experiences=n_experiences,
            memory_size=memory_size,
            memory_strength=memory_strength,
            proj_interval=proj_interval,
            patterns_per_exp=patterns_per_exp,
            proj_metric=proj_metric,
        )
        self.pgd_iterations  = pgd_iterations
        self.lr              = lr
        self.use_adaptive_lr = use_adaptive_lr
        self.use_warm_start  = use_warm_start
        self.memory_x        = {}
        self.memory_y        = {}
        self.memory_tid      = {}
        self.G               = torch.empty(0)
        self.GGT             = torch.empty(0)
        self.v               = None

    def _reset_memory(self, strategy):
        self.memory_x = {}
        self.memory_y = {}
        self.memory_tid = {}
    def _has_memory(self) -> bool:
        return bool(self.memory_x)

    def _compute_reference_gradients(self, strategy) -> Tensor:
        G_list = []
        for t in range(strategy.clock.train_exp_counter):
            strategy.optimizer.zero_grad()
            xref = self.memory_x[t].to(strategy.device)
            yref = self.memory_y[t].to(strategy.device)
            out = avalanche_forward(strategy.model, xref, self.memory_tid[t])
            loss = strategy._criterion(out, yref)
            loss.backward()
            flat = [
                (p.grad.detach().clone().flatten() if p.grad is not None else torch.zeros(p.numel(), device=strategy.device))
                for p in strategy.model.parameters()
            ]
            G_list.append(torch.cat(flat, dim=0))
        self.G = torch.stack(G_list, dim=0)
        if self.use_adaptive_lr:
            self.GGT = self.G @ self.G.T
            L = torch.linalg.eigvalsh(self.GGT).max()
            self.lr = 1.0 / (L + 1e-6)
        return self.G
    def _solve_projection(self, g, reference, memory_strength):
        return self._solve_projection_pgd(
            v=self.v,
            t=reference.shape[0],
            dev=g.device,
            G=self.G,
            g=g,
            GGT=self.GGT,
            I=self.pgd_iterations,
            lr=self.lr,
            memory_strength=memory_strength,
            use_adaptive_lr=self.use_adaptive_lr,
            use_warm_start=self.use_warm_start
        )
    def _solve_projection_pgd(self, v, t, dev, G, g, GGT, I, lr, memory_strength, use_adaptive_lr, use_warm_start):
        '''
        theory: v* <- 0-vector
        gradF w/ respect to v: G * (transpose(G) * v) + G * g
        new-v_star <- old-v_star - alpha * gradF
        new-v_star <- max[0-vector, v]
        '''
        #print(f"[DEBUG] Solving projection with parameters: t={t}, dev={dev}, I={I}, lr={lr}, memory_strength={memory_strength}, use_adaptive_lr={use_adaptive_lr}, use_warm_start={use_warm_start}")
        if v is None or v.shape[0] != t or not use_warm_start:
            v = torch.zeros(t, device=dev)

        z = torch.full_like(v, memory_strength, device=dev)

        Gg = torch.mv(G, g)
        for _ in range(I):
            if not use_adaptive_lr:
                temp = torch.mv(G.T, v) # s ∈ n x 1
                full_product = torch.mv(G, temp)
            else:
                full_product = torch.mv(GGT, v)
            gradF = full_product + Gg
            v -= lr * gradF
            v = torch.max(v, z)
        g_proj = torch.mv(G.T, v) + g
        
        if use_warm_start:
            self.v = v
        return g_proj.to(dev)
    
    def _update_memory(self, strategy):
        ds = strategy.experience.dataset
        t = strategy.clock.train_exp_counter
        batch_size = strategy.train_mb_size
        collate_fn = getattr(ds, 'collate_fn', None)
        loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        tot = 0
        for mb in loader:
            x, y, tid = mb[0], mb[1], mb[-1]
            bsz = x.size(0)

            if tot + bsz <= self.patterns_per_exp:
                if t not in self.memory_x:
                    self.memory_x[t] = x.clone()
                    self.memory_y[t] = y.clone()
                    self.memory_tid[t] = tid.clone()
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x.clone()), dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], y.clone()), dim=0)
                    self.memory_tid[t] = torch.cat((self.memory_tid[t], tid.clone()), dim=0)
                    mem_diff = self.memory_x[t].size(0) - self.memory_size
                    if mem_diff > 0:
                        self.memory_x[t] = self.memory_x[t][mem_diff:]
                        self.memory_y[t] = self.memory_y[t][mem_diff:]
                        self.memory_tid[t] = self.memory_tid[t][mem_diff:]
            else:
                diff = self.patterns_per_exp - tot
                if t not in self.memory_x:
                    self.memory_x[t] = x[:diff].clone()
                    self.memory_y[t] = y[:diff].clone()
                    self.memory_tid[t] = tid[:diff].clone()
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x[:diff].clone()), dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], y[:diff].clone()), dim=0)
                    self.memory_tid[t] = torch.cat((self.memory_tid[t], tid[:diff].clone()), dim=0)
                    mem_diff = self.memory_x[t].size(0) - self.memory_size
                    if mem_diff > 0:
                        self.memory_x[t] = self.memory_x[t][mem_diff:]
                        self.memory_y[t] = self.memory_y[t][mem_diff:]
                        self.memory_tid[t] = self.memory_tid[t][mem_diff:]
                break  

            tot += bsz

    def _should_project(self, g, mem_strength):
        return (torch.mv(self.reference, g) < -mem_strength).any()
