from abc import abstractmethod
from typing import Any
import torch
from torch import Tensor
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from . import core 

class BaseGEMPlugin(SupervisedPlugin):

    def __init__(
        self,
        *,
        memory_strength: float,
        proj_interval: int,
        patterns_per_exp: int,
        n_experiences: int,
        memory_size: int,
        proj_metric,
    ):
        super().__init__()
        self.memory_strength      = memory_strength
        self.proj_interval        = proj_interval
        self.patterns_per_exp     = patterns_per_exp
        self.projection_iteration = 0
        self.proj_metric          = proj_metric
        self.memory_size          = memory_size // n_experiences
        
        if self.proj_metric:
            print("GEM: using projection metric")
            print(f"Projection interval: {self.proj_interval}")
            
        # To be set by derived classes:
        # GEM: matrix of gradient references
        # AGEM: single averaged gradient reference
        self.reference : Tensor = None
        
    def before_training_iteration(self, strategy, *args, **kwargs):
        # batch_x, batch_y = strategy.mb_x, strategy.mb_y
        # for i in range(50):
        #     strategy.model.train()
        #     strategy.optimizer.zero_grad()
        #     out   = strategy.model(batch_x)
        #     loss  = strategy._criterion(out, batch_y)
        #     loss.backward()
        #     strategy.optimizer.step()
        #     print(f"Step {i:2d} — loss: {loss.item():.4f}")
        # raise ValueError
        if not self._has_memory():
            return
        self.reference = self._compute_reference_gradients(strategy)
        
        strategy.optimizer.zero_grad()

    @torch.no_grad()
    def after_backward(self, strategy, *args, **kwargs):
        if not self._has_memory():
            return
    
        g_list = [
            (p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=strategy.device))
            for p in strategy.model.parameters()
        ]
        g = torch.cat(g_list, dim=0)

        do_proj = (
            self.projection_iteration % self.proj_interval == 0
            and self._should_project(g, self.memory_strength) 
        )
        
        self.projection_iteration += 1
        
        if not do_proj:
            return
        g_proj, elapsed = core.time_projection(
            self._solve_projection,
            g=g,
            reference=self.reference,
            memory_strength=self.memory_strength
        )
        if self.proj_metric:
            self.proj_metric.elapsed += elapsed
            
        offset = 0
        for p in strategy.model.parameters():
            n = p.numel()
            if p.grad is not None:
                p.grad.copy_(g_proj[offset : offset + n].view_as(p))
            offset += n
        assert offset == g_proj.numel(), f"g_proj size {g_proj.numel()} does not match model g size {offset}"
    
    def after_training_exp(self, strategy, *args, **kwargs):
        self._update_memory(strategy)

    @abstractmethod
    def _reset_memory(self, strategy):
        pass
    @abstractmethod
    def _has_memory(self):
        pass
    @abstractmethod
    def _update_memory(self, strategy):
        pass
    @abstractmethod
    def _should_project(self, g, mem_strength) -> bool:
        pass    
    @abstractmethod
    def _solve_projection(self, **kwargs) -> Tensor:
        pass
    @abstractmethod
    def _compute_reference_gradients(self, strategy) -> torch.Tensor:
        pass

    