from typing import Dict
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
import time

class ODCAGEMPlugin(SupervisedPlugin):
    """
    GEM via Online Dual Coordinate‐Ascent.
    Projects the current minibatch gradient so that its dot‐products
    with all stored past‐task gradients remain non‐negative,
    updating one dual coordinate (or a small number of them) per batch.
    """

    def __init__(
        self,
        patterns_per_exp: int,
        memory_strength: float,
        coord_iterations: int,
        proj_metric
    ):
        """
        :param patterns_per_exp: # of samples to store per experience
        :param memory_strength: constant offset added to the dual gradient
                                to bias towards backward transfer
        :param coord_iterations: # of coordinate updates per new batch
        """
        super().__init__()
        self.patterns_per_experience = int(patterns_per_exp)
        self.memory_strength = memory_strength
        self.coord_iterations = coord_iterations

        # replay memory
        self.memory_x: Dict[int, Tensor] = {}
        self.memory_y: Dict[int, Tensor] = {}
        self.memory_tid: Dict[int, Tensor] = {}

        # gradient matrix (T × d) and its Gram
        self.G: Tensor = torch.empty(0)
        self.GGT: Tensor = torch.empty(0)
        self.diagGGT: Tensor = torch.empty(0)

        # dual variables (length T) and coordinate pointer
        self.alpha: Tensor = None
        self._next_coord = 0

        # projection‐time metric
        self.proj_metric = proj_metric

    def before_training_iteration(self, strategy, **kwargs):
        # build G, GGT, diag(GGT) once per task
        t_count = strategy.clock.train_exp_counter
        if t_count > 0:
            grads = []
            strategy.model.train()
            for t in range(t_count):
                strategy.optimizer.zero_grad()
                xref = self.memory_x[t].to(strategy.device)
                yref = self.memory_y[t].to(strategy.device)
                out = avalanche_forward(strategy.model, xref, self.memory_tid[t])
                loss = strategy._criterion(out, yref)
                loss.backward()
                grads.append(torch.cat([
                    (p.grad.flatten() if p.grad is not None
                     else torch.zeros(p.numel(), device=strategy.device))
                    for p in strategy.model.parameters()
                ], dim=0))
            self.G = torch.stack(grads)                 # (T, d)
            self.GGT = self.G @ self.G.T                # (T, T)
            self.diagGGT = torch.diag(self.GGT)         # (T,)
            # init duals if needed
            if (self.alpha is None) or (self.alpha.shape[0] != t_count):
                self.alpha = torch.zeros(t_count, device=strategy.device)
                self._next_coord = 0

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        # only project if we have past tasks
        t_count = strategy.clock.train_exp_counter
        if t_count == 0:
            return

        # flatten current gradient
        g = torch.cat([
            (p.grad.flatten() if p.grad is not None
             else torch.zeros(p.numel(), device=strategy.device))
            for p in strategy.model.parameters()
        ], dim=0)

        # check if any constraint violated
        viol = (self.G @ g < 0).any().item()
        if not viol:
            return

        # perform coord.‐ascent updates
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        Gg = self.G @ g                         # precompute once
        for _ in range(self.coord_iterations):
            i = self._next_coord
            # full dual‐gradient at coord i
            grad_i = (self.GGT[i] @ self.alpha) + Gg[i]
            # add memory_strength bias
            grad_i += self.memory_strength
            # coordinate‐descent step on F(α)=½αᵀGGTα + (Gg)ᵀα
            self.alpha[i] = torch.clamp(
                self.alpha[i] - grad_i / (self.diagGGT[i] + 1e-12),
                min=0.0
            )
            # advance cyclic coordinate
            self._next_coord = (i + 1) % t_count
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.proj_metric.elapsed += t1 - t0

        # reconstruct projected gradient and write back
        g_proj = g + self.G.T @ self.alpha
        ptr = 0
        for p in strategy.model.parameters():
            n = p.numel()
            if p.grad is not None:
                p.grad.copy_(g_proj[ptr:ptr + n].view(p.size()))
            ptr += n
        assert ptr == g_proj.numel(), "Mismatch in projected‐gradient size"

    @torch.no_grad()
    def after_training_exp(self, strategy, **kwargs):
        # refill memory from this experience
        self._update_memory(
            strategy.experience.dataset,
            strategy.clock.train_exp_counter,
            strategy.train_mb_size
        )

    @torch.no_grad()
    def _update_memory(self, dataset, t, batch_size):
        collate = getattr(dataset, "collate_fn", None)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate)
        stored = 0
        for minib in loader:
            x, y, tid = minib[0], minib[1], minib[-1]
            take = min(x.size(0), self.patterns_per_experience - stored)
            if t not in self.memory_x:
                self.memory_x[t] = x[:take].clone()
                self.memory_y[t] = y[:take].clone()
                self.memory_tid[t] = tid[:take].clone()
            else:
                self.memory_x[t] = torch.cat((self.memory_x[t], x[:take]), dim=0)
                self.memory_y[t] = torch.cat((self.memory_y[t], y[:take]), dim=0)
                self.memory_tid[t] = torch.cat((self.memory_tid[t], tid[:take]), dim=0)
            stored += take
            if stored >= self.patterns_per_experience:
                break
