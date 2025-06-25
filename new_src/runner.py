# benchmark each command
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
    # IGEM
    "sgd_iterations": 3,
    # AGEM
    "sample_size" : -1, # AGEM placeholder,
    # N_ITER
    "projection_iteration": [1, 25, 100, 500],
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
    "output_dir": "./new_src/debug_results/minitest",
    "cuda" : 0,
    
    # N_ITER
    "projection_iteration": [2, 100],
    # GEM
    "memory_size" : 10, 
    #AGEM
    "sample_size" : -1,
}

'''
example runs (file + options)
> python3 new_src/runner.py --strategy gem --benchmark permuted-mnist --model mlp --plugin warm_n_iter

> python3 new_src/runner.py --strategy agem --benchmark permuted-mnist --model mlp --plugin agem_n_iter
'''
