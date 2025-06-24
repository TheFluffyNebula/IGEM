import argparse
import subprocess
import shlex

BENCHMARK_MODEL_MAP = {
    "permuted-mnist" : ["mlp", "multimodal"],
    #"rotated-mnist" :  ["mlp", "multimodal"],
    "cifar100" :       ["resnet18"]
}
STRAT_MODEL_MAP = {
    #"single":       ["mlp", "resnet18"],
    #"independent":  ["mlp", "resnet18"],
   # "multimodal":   ["multimodal"],
   # "icarl":        ["resnet18"],
   # "ewc":          ["mlp", "resnet18"],
    "gem":          ["mlp", "resnet18"],
    "agem":         ["mlp", "resnet18"],
}
# BENCHMARK_MODEL_MAP[STRAT_BENCHMARK_MAP["single"][0]]].intersect(STRAT_MODEL_MAP["single"])
STRAT_BENCHMARK_MAP = {
   # "single":       ["permuted-mnist", "rotated-mnist", "cifar100"],
   # "independent":  [ "permuted-mnist", "rotated-mnist", "cifar100"],
   # "multimodal":   ["permuted-mnist", "rotated-mnist"],
   # "icarl":        ["cifar100"],
   # "ewc":          ["permuted-mnist", "rotated-mnist", "cifar100"],
    "gem":          ["permuted-mnist"], # "cifar100"
    "agem":         ["permuted-mnist"], # "cifar100"
}

STRAT_PLUGIN_MAP = {
   # "single":       [None],   
   # "independent":  [None],
   # "multimodal":   [None],
   # "icarl":        [None],
   # "ewc":          [None],  
   "gem":           ["gem", "sgem_n_iter", "blr_n_iter", "warm_n_iter"],# "sgem", "gem","sgem_best_learning_rate", ["sketch_gem", "odcagem"], "n_iter", "blr_n_iter"
    "agem":         ["agem_n_iter"], #"agem",
}
STRAT_MEMSIZE_MAP = {
    "gem" : [200, 1280, 2560, 5120]
}
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--debug",
        action='store_true'
    )
    args = p.parse_args()    
    for strategy in STRAT_BENCHMARK_MAP.keys():
        for benchmark in STRAT_BENCHMARK_MAP[strategy]:
            models = list(set(BENCHMARK_MODEL_MAP[benchmark]).intersection(set(STRAT_MODEL_MAP[strategy])))
            for model in models:
                for plugin in STRAT_PLUGIN_MAP[strategy]:
                    # print(f"\nRunning {strategy} on {benchmark} with {model} and plugin {plugin}")
                    # run_experiment(strategy, benchmark, model, plugin, args.debug)
                    cmd = [
                        # "torchrun",
                        # "--nnodes=1",
                        # "--nproc_per_node=8",
                        # # If you’re using explicit rendezvous:
                        # "--rdzv_backend=c10d",
                        # "--rdzv_endpoint=127.0.0.1:29500",
                        # "--rdzv_id=sgem_experiment",
                        "python3",
                        "src/main.py",
                        "--strategy",  strategy,
                        "--benchmark", benchmark,
                        "--model",     model,
                    ]
                    if plugin is not None:
                        cmd += ["--plugin", plugin]
                    if args.debug:
                        cmd.append("--debug")

                    # Nicely print the command
                    print("\n> " + " ".join(shlex.quote(x) for x in cmd))
                    # Run it (will block until this torchrun exits)
                    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()