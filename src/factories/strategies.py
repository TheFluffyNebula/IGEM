from avalanche.training.supervised import GEM, AGEM, ICaRL, EWC, Naive
import inspect
from strategies import IndependentStrategy, MultimodalStrategy
from functools import wraps
import torch.nn as nn

_registry: dict[str, type] = {}

def register_strategy(name: str):
    def decorator(cls):
        _registry[name] = cls
        return cls
    return decorator

@register_strategy("gem")
@wraps(GEM)
def GEMFactory(**kwargs):
    return GEM(**kwargs)

@register_strategy("agem")
@wraps(AGEM)
def AGEMFactory(**kwargs):
    return AGEM(**kwargs)

@register_strategy("icarl")
@wraps(ICaRL)
def ICaRLFactory(**kwargs):
    num_classes = 100  # for CIFAR-100
    from factories.models import make_model
    model = make_model("resnet18", num_classes=num_classes)
    classifier = model.linear
    model.linear = nn.Identity()
    return ICaRL(feature_extractor=model,
                 classifier=classifier,
                 fixed_memory=True,
                 buffer_transform=None,
                 **kwargs)

@register_strategy("ewc")
@wraps(EWC)
def EWCFactory(**kwargs):
    print(kwargs["ewc_lambda"])
    return EWC(**kwargs)

@register_strategy("single")
@wraps(Naive)
def SingleFactory(**kwargs):
    return Naive(**kwargs)

@register_strategy("independent")
@wraps(Naive)
def IndependentFactory(**kwargs):
    finetune = kwargs.pop('finetune', True)
    return IndependentStrategy(Naive,finetune=finetune, **kwargs)

@register_strategy("multimodal")
@wraps(Naive)
def MultimodalFactory(**kwargs):
    return Naive(**kwargs)


def make_strategy(name: str, **kwargs):
    """
    :name: str,
    :model: torch.nn.Module,
    :optimizer: torch.optim.Optimizer,
    :criterion: torch.nn.Module,
    :eval_plugin: EvaluationPlugin,
    :device: torch.device,
    :patterns_per_exp: int,
    :memory_strength: float,
    :train_mb_size: int,
    :train_epochs: int,
    :eval_mb_size: int,
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
    print(f'train_mb_size: {filtered_kwargs["train_mb_size"]}')
    return factory(**filtered_kwargs)

