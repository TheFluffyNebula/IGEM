from util import parse_args
from eval import BenchmarkRunner
import argparse
DEFAULT_ARGS = {
    "seed": 0,
    "train_mb_size": 32,      
    "eval_mb_size": 1000,
    "n_experiences": 10,      
    "train_epochs": 1,    
    "lr": 0.1,               
    "momentum": 0.0,
    "patterns_per_exp": 256,   
    "memory_strength": 0.5,
    "output_dir": "./eval_results",
    "cuda": 0,
    
    # GEM
    "memory_size" : 10, 
    # N_ITER
    "projection_iteration": [1, 25, 100, 500], 
    # IGEM
    "sgd_iterations": 3,  
    # EWC
    "ewc_lambda": -1, # placeholder,
    # AGEM
    "sample_size" : -1, # AGEM placeholder,
    # ODCA
    "coord_iterations": 4
}
DEBUG_ARGS = {
    "seed": 0,
    "train_mb_size": 32,
    "eval_mb_size": 1000,
    "n_experiences": 3,
    "train_epochs": 1,
    "lr": 0.01,
    "momentum": 0.0,
    "patterns_per_exp": 10,
    "memory_strength": 0.5,
    "output_dir": "./debug_results/minitest",
    "cuda" : 0,
    
    # N_ITER
    "projection_iteration": [2, 100],  
    # GEM
    "memory_size" : 10, 
    #EWC
    "ewc_lambda": -1,
    #AGEM
    "sample_size" : -1,
    #IGEM
    "sgd_iterations" : 1,
    # ODCA
    "coord_iterations": 4,
}
def make_args(strategy, benchmark, model, plugin, result_filename=None, DEBUG=False):
    base = DEBUG_ARGS if DEBUG else DEFAULT_ARGS
    args = argparse.Namespace(**base)
    args.strategy = strategy
    args.benchmark = benchmark
    args.model = model
    args.plugin = plugin
    args.result_filename = result_filename

    if strategy == "gem":
        args.patterns_per_exp = 256
        args.memory_strength = 0.5     
        args.lr = 0.1                  
    elif strategy == "agem":
        if benchmark == "permuted-mnist":
            args.patterns_per_exp = 250
            args.sample_size      = 256
        elif benchmark == "cifar100":
            args.patterns_per_exp = 65
            args.sample_size      = 1300
            
    return args

def run_experiment(strategy, benchmark, model, plugin,  DEBUG=False):

    base_args = make_args(
        strategy=strategy,
        benchmark=benchmark,
        model=model,
        plugin=plugin,
        DEBUG=DEBUG
    )

    projection_iterations = base_args.projection_iteration
    if not isinstance(projection_iterations, list):
        projection_iterations = [projection_iterations]
    if "iter" not in plugin:
        projection_iterations = [1]
    for proj_iter in projection_iterations:
        base_args.projection_iteration = proj_iter
        base_args.result_filename = f"{strategy}_{benchmark}_{model}_{plugin}_{proj_iter}.csv"

        print(f"\nRunning with projection_iteration = {proj_iter}")
        runner = BenchmarkRunner(base_args)
        runner.setup_distributed()
        runner.setup_device_and_seed()
        runner.prepare_data()
        runner.build_model_and_strategy(DEBUG)
        runner.run()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single Avalanche benchmark experiment under DDP"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Name of the continual learning strategy to use"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Identifier of the benchmark/dataset to run"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model architecture to instantiate"
    )
    parser.add_argument(
        "--plugin",
        type=str,
        required=False,
        help="Name of the Avalanche plugin to attach"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (more verbose logging, no DDP wrap, etc.)"
    )
    return parser.parse_args()

def run():
    args = parse_args()
    
    print(f"Running {args.strategy} on {args.benchmark} with {args.model} and {args.plugin}")
    run_experiment(
        strategy=args.strategy,
        benchmark=args.benchmark,
        model=args.model,
        plugin=getattr(args, "plugin", None),
        DEBUG=args.debug
    )
    
if __name__ == "__main__":
    run()
