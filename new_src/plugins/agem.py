import time
import warnings
import random
from typing import Any, Iterator, List, Optional
import torch
from torch import Tensor

from avalanche.benchmarks.utils import _make_taskaware_classification_dataset
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_loader import (
    GroupBalancedInfiniteDataLoader,
)
from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from .base_gem import BaseGEMPlugin
from . import core
class AGEMPlugin(BaseGEMPlugin):
    """Average GEM (A-GEM) concrete implementation."""
    def __init__(
        self,
        patterns_per_exp: int,
        sample_size: int,
        memory_strength: float,
        proj_interval: int,
        proj_metric: Any,
    ):
        print("=======Using AGEM plugin======")
        super().__init__(
            memory_strength=memory_strength,
            proj_interval=proj_interval,
            patterns_per_exp=patterns_per_exp,
            proj_metric=proj_metric,
        )
        self.sample_size = sample_size
        self.buffers: list = []
        self.buffer_dliter = iter([])

    def _has_memory(self) -> bool:
        return len(self.buffers) > 0
        
    @torch.no_grad()
    def _update_memory(self, strategy, **kwargs):
        dataset = strategy.experience.dataset
        num_workers = 0
        if num_workers > 0:
            warnings.warn(
                "Num workers > 0 is known to cause heavy" "slowdowns in AGEM."
            )
        removed_els = len(dataset) - self.patterns_per_exp
        if removed_els > 0:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = dataset.subset(indices[: self.patterns_per_exp])

        self.buffers.append(dataset)

        persistent_workers = num_workers > 0
        self.buffer_dataloader = GroupBalancedInfiniteDataLoader(
            self.buffers,
            batch_size = max(1, self.sample_size // len(self.buffers)),
            num_workers=num_workers,w3
            pin_memory=False,
            persistent_workers=persistent_workers,
        )
        self.buffer_dliter = iter(self.buffer_dataloader)
    
    def _should_project(self, g, mem_strength) -> bool:
        margin = mem_strength *  torch.dot(self.reference, self.reference)
        return torch.dot(self.reference, g) < -margin
    
    def _solve_projection(self, g: Tensor, reference: Tensor, memory_strength: float):
        print("Projecting AGEM gradient")
        sq = torch.dot(reference, reference)
        dotg = torch.dot(g, reference) + memory_strength * sq
        alpha = dotg / sq
        return (g - alpha * reference).to(g.device)

    def _compute_reference_gradients(self, strategy) -> Tensor:
        # Sample one minibatch
        print(self.buffers)
        batch = next(iter(self.buffer_dliter))
        try:
            xref, yref, tid = batch
        except ValueError:
            xref, yref = batch
            tid = torch.zeros_like(yref)
        xref, yref = xref.to(strategy.device), yref.to(strategy.device)
        out = avalanche_forward(strategy.model, xref, tid)
        loss = strategy._criterion(out, yref)
        loss.backward()

        # Flatten into vector
        flat = [
            (p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=strategy.device))
            for p in strategy.model.parameters()
        ]
        return torch.cat(flat, dim=0)
