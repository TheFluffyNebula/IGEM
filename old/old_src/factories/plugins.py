from avalanche.training import GEM, AGEM, ICaRL, EWC
from plugins import (
    SGEMPlugin, 
    SGEMBestLRPlugin, 
    SketchGEMPlugin, 
    GEMPlusPlugin, 
    AGEMPlusPlugin, 
    AGEMPlusNITPlugin, 
    SGEMWarmNITPlugin, 
    ODCAGEMPlugin,
    SGEMBestLRNITPlugin,
    SGEMNITPlugin,
    )
import inspect
from typing import Callable
from functools import wraps
_registry: dict[str, Callable] = {}

def register_plugin(name: str):
    def decorator(cls):
        _registry[name] = cls
        return cls
    return decorator

# @register_plugin("gem")
# class GEMFactory(GEM):
#     pass

# @register_plugin("agem")
# class AGEMFactory(AGEM):
#     pass

# @register_plugin("icarl")
# class ICaRLFactory(ICaRL):
#     pass

# @register_plugin("ewc")
# class EWCFactory(EWC):
#     pass

@register_plugin("sgem")
@wraps(SGEMPlugin)
def SGEMFactory(**kwargs):
    return SGEMPlugin(**kwargs)

@register_plugin("sgem_best_learning_rate")
@wraps(SGEMBestLRPlugin)
def SGEMBestLRPFactory(**kwargs):
    return SGEMBestLRPlugin(**kwargs)

@register_plugin("sketch_gem")
@wraps(SketchGEMPlugin)
def SGEMSketchFactory(**kwargs):
    return SketchGEMPlugin(**kwargs)

@register_plugin("agem")
@wraps(AGEMPlusPlugin)
def AGEMPlusFactory(**kwargs):
    return AGEMPlusPlugin(**kwargs)

@register_plugin("agem_n_iter")
@wraps(AGEMPlusNITPlugin)
def AGEMPlusNITPluginFactory(**kwargs):
    return AGEMPlusNITPlugin(**kwargs)

@register_plugin("gem")
@wraps(GEMPlusPlugin)
def GEMPlusFactory(**kwargs):
    return GEMPlusPlugin(**kwargs)

@register_plugin("warm_n_iter")
@wraps(SGEMWarmNITPlugin)
def SGEMWARMFactory(**kwargs):
    return SGEMWarmNITPlugin(**kwargs)

@register_plugin("odcagem")
@wraps(ODCAGEMPlugin)
def ODCAGEMFactory(**kwargs):
    return ODCAGEMPlugin(**kwargs)

@register_plugin("sgem_n_iter")
@wraps(SGEMNITPlugin)
def SGEMNITFactory(**kwargs):
    return SGEMNITPlugin(**kwargs)

@register_plugin("blr_n_iter")
@wraps(SGEMBestLRNITPlugin)
def SGEMBestLRNITFactory(**kwargs):
    return SGEMBestLRNITPlugin(**kwargs)

def make_plugin(name: str, **kwargs):
    """
    :patterns_per_experience: int
    :memory_strength: float
    """
    if not name:
        return None
    try:
        factory = _registry[name]
    except KeyError:
        raise ValueError(f"Unknown Plugin {name!r}")

    real = inspect.unwrap(factory)
    sig = inspect.signature(real)
    # Keep only those entries in kwargs that the factory actually accepts
    filtered_kwargs = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters
    }

    return factory(**filtered_kwargs)