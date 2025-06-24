from avalanche.models import SimpleMLP, SlimResNet18, as_multitask
from typing import Callable
import inspect
from functools import wraps
from avalanche.models import MultiTaskModule
_registry: dict[str, Callable] = {}

def register_model(name: str):
    def decorator(cls):
        _registry[name] = cls
        return cls
    return decorator

@register_model("mlp")
@wraps(SimpleMLP)
def SimpleMLPFactory(**kwargs):
    return SimpleMLP(hidden_size=100, hidden_layers=2, **kwargs)

@register_model("multimodal")
@wraps(MultiTaskModule)
def MultiModalMLPFactory(**kwargs):
    return as_multitask(SimpleMLP(**kwargs), "classifier")

@register_model("resnet18")
@wraps(SlimResNet18)
def ResNetFactory(**kwargs):
    return SlimResNet18(**kwargs)

def make_model(name: str, **kwargs):
    """
    :num_classes: int
    """
    try:
        factory = _registry[name]
    except KeyError:
        raise ValueError(f"Unknown Strategy {name!r}")

    # Grab the factory’s signature
    real = inspect.unwrap(factory)
    sig = inspect.signature(real)
    # Keep only those entries in kwargs that the factory actually accepts
    # ─── ALIAS num_classes → n_classes ────────────────────────────────
    # SlimResNet18 (and any other fn that uses 'n_classes') will
    # accept 'n_classes', not 'num_classes', so copy it over if present.
    if "num_classes" in kwargs and "nclasses" in sig.parameters:
       kwargs["nclasses"] = kwargs.pop("num_classes")
   # ─────────────────────────────────────────────────────────────────────
    filtered_kwargs = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters
    }

    return factory(**filtered_kwargs)

