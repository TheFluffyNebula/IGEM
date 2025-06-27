from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from base_gem import IGEMPlugin
import core  

class PseudoIGEMPlugin(IGEMPlugin):
    def __init__(self,
        patterns_per_exp: int,
        pgd_iterations: int,
        lr: float,
        use_adaptive_lr: bool,
        use_warm_start: bool,
        memory_strength: float,
        cluster_strength: float,
        proj_interval: int,
        proj_metric: Any,
        
        ):
        super.__init__(
            patterns_per_exp=patterns_per_exp,
            pgd_iterations=pgd_iterations,
            lr=lr,
            use_adaptive_lr=use_adaptive_lr,
            use_warm_start=use_warm_start,
            memory_strength=memory_strength,
            proj_interval=proj_interval,
            proj_metric=proj_metric
        )
        self.cluster_strength = cluster_strength
        

        