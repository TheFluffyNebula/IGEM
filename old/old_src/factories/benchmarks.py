from typing import Callable
from avalanche.benchmarks.classic import (
        PermutedMNIST,
        RotatedMNIST,
        SplitCIFAR100
        )
from avalanche.benchmarks import NCScenario
import inspect
from functools import wraps
_registry: dict[str, Callable]= {}


def register_benchmark(name: str):
    def decorator(cls):
        _registry[name] = cls
        return cls
    return decorator


@register_benchmark("permuted-mnist")
@wraps(NCScenario)
def PermutedMNISTFactory(n_experiences, seed, **kwargs):
    return PermutedMNIST(n_experiences, seed=seed,**kwargs)

@register_benchmark("rotated-mnist")
@wraps(NCScenario)
def RotatedMNISTFactory(n_experiences, seed, **kwargs):
    return RotatedMNIST(n_experiences, seed=seed,rotations_list=[i * 9 for i in range(n_experiences)], **kwargs)

@register_benchmark("cifar100")
@wraps(NCScenario)
def SplitCIFAR100Factory(n_experiences, seed, **kwargs):
    return SplitCIFAR100(n_experiences, seed=seed,return_task_id=True, **kwargs)


def make_benchmark(name: str, n_experiences: int, seed: int, **kwargs):
    """
    :n_experiences: int
    :seed: int
    """
    try:
        factory = _registry[name]
    except KeyError:
        raise ValueError(f"Unknown Strategy {name!r}")
    
    real = inspect.unwrap(factory)
    sig = inspect.signature(real)
    # Keep only those entries in kwargs that the factory actually accepts
    filtered_kwargs = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters
    }

    return factory(n_experiences, seed=seed, **filtered_kwargs)


