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
        proj_metric: Any = None,
    ):
        super().__init__()
        self.memory_strength      = memory_strength
        self.proj_interval        = proj_interval
        self.patterns_per_exp     = patterns_per_exp
        self.projection_iteration = 0
        self.proj_metric          = proj_metric
        
        # To be set by derived classes:
        # GEM: matrix of gradient references
        # AGEM: single averaged gradient reference
        self.reference : Tensor = None
        
    def before_training_iteration(self, strategy, **kwargs):
        if not self._has_memory():
            return
        
        # reset projection iteration every task
        self.projection_iteration = 0
        strategy.model.train()
        strategy.optimizer.zero_grad()
        
        # delegate method: draw from memory, forward/back pass, flatten grads
        self.reference = self._compute_reference_gradients(strategy)
        
        strategy.optimizer.zero_grad()

    @torch.no_grad()
    def after_backward(self, strategy, *args, **kwargs):
        if not self._has_memory():
            return
    
        # gather model gradients into a single vector
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
        print(f"p_iter: {self.projection_iteration}")
        
        if not do_proj:
            return
        # calculate projected gradient and time it
        g_proj, elapsed = core.time_projection(
            self._solve_projection,
            g=g,
            reference=self.reference,
            memory_strength=self.memory_strength
        )
        # update projection overhead metric
        if self.proj_metric:
            self.proj_metric.elapsed += elapsed
        
        # write back into params
        offset = 0
        for p in strategy.model.parameters():
            n = p.numel()
            if p.grad is not None:
                p.grad.copy_(g_proj[offset : offset + n].view_as(p))
            offset += n
        assert offset == g_proj.numel(), f"g_proj size {g_proj.numel()} does not match model g size {offset}"
    
    def after_training_exp(self, strategy, **kwargs):
        # Get the stored experience from the strategy
        if hasattr(strategy, '_current_experience'):
            self._update_memory(strategy._current_experience)
        else:
            print("Warning: No current experience found in strategy")
    
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

    