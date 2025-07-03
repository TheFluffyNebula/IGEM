import inspect
from avalanche.benchmarks.classic import PermutedMNIST, SplitCIFAR100
from avalanche.models import SimpleMLP, SlimResNet18
from plugins import AGEMPlugin, GEMPlugin, IGEMPlugin
from eval.mmlu_benchmark import make_mmlu_benchmark

def make_benchmark(name: str, n_experiences: int, seed: int):
    if name == "permuted-mnist":
        return PermutedMNIST(n_experiences=n_experiences, seed=seed)
    elif name == "cifar100":
        return SplitCIFAR100(n_experiences=n_experiences, seed=seed)
    elif name == "mmlu-cl":
        return make_mmlu_benchmark(mmlu_root="new_src/data/mmlu", n_experiences=n_experiences, seed=seed)
    else:
        raise ValueError(f"Unknown benchmark: {name}")

def make_model(name: str, **kwargs):
    if name == "mlp":
        factory = SimpleMLP
    elif name == "resnet18":
        factory = SlimResNet18
    elif name == "gpt2":
        print("model gpt2 todo")
    else:
        raise ValueError(f"Unknown model name: {name!r}")

    sig = inspect.signature(factory)

    # Handle aliases: num_classes → n_classes or nclasses
    if "num_classes" in kwargs:
        if "n_classes" in sig.parameters:
            kwargs["n_classes"] = kwargs.pop("num_classes")
        elif "nclasses" in sig.parameters:
            kwargs["nclasses"] = kwargs.pop("num_classes")

    # Filter kwargs to only what the factory accepts
    filtered_kwargs = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters
    }

    return factory(**filtered_kwargs)

def make_plugin(plugin_name: str, **kwargs):
    if plugin_name == "gem":
        return GEMPlugin(
            patterns_per_exp=kwargs["patterns_per_exp"],
            memory_strength=kwargs["memory_strength"],
            proj_interval=kwargs["proj_interval"],
            proj_metric=kwargs["proj_metric"]
        )

    elif plugin_name == "igem":
        return IGEMPlugin(
            patterns_per_exp=kwargs["patterns_per_exp"],
            memory_strength=kwargs["memory_strength"],
            proj_interval=kwargs["proj_interval"],
            proj_metric=kwargs["proj_metric"],
            pgd_iterations=kwargs["pgd_iterations"],
            use_adaptive_lr=kwargs.get("adaptive_lr", False),
            use_warm_start=kwargs.get("warm_start", False),
            lr=kwargs["lr"]
        )

    elif plugin_name == "agem":
        return AGEMPlugin(
            sample_size=kwargs["sample_size"]
        )

    else:
        raise ValueError(f"Unknown plugin type: {plugin_name}")
