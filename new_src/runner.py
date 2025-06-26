class Runner:
    def __init__(self, benchmark, strategy, 
            seed, proj_interval,
            train_mb_size, eval_mb_size,
            n_experiences, train_epochs,
            lr, momentum, patterns_per_exp,
            memory_strength, output_dir, cuda, 
            **kwargs
        ):
        self.benchmark = benchmark
        self.strategy = strategy
        self.seed = seed
        self.proj_interval = proj_interval
        self.train_mb_size = train_mb_size
        self.eval_mb_size = eval_mb_size
        self.n_experiences = n_experiences
        self.train_epochs = train_epochs
        self.lr = lr
        self.momentum = momentum
        self.patterns_per_exp = patterns_per_exp
        self.memory_strength = memory_strength
        self.output_dir = output_dir
        self.cuda = cuda

        # Store any additional args as attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
    


'''
example runs (file + options)
> python3 new_src/runner.py --strategy gem --benchmark permuted-mnist --model mlp --plugin warm_n_iter

> python3 new_src/runner.py --strategy agem --benchmark permuted-mnist --model mlp --plugin agem_n_iter
'''
