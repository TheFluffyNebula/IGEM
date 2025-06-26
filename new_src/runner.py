class Runner:
    # explicitly pass all keywords (replace strategy_keywords)
    def __init__(self, benchmark, strategy, strategy_keywords):
        self.benchmark = benchmark
        self.strategy = strategy
        self.strategy_keywords = strategy_keywords
    


'''
example runs (file + options)
> python3 new_src/runner.py --strategy gem --benchmark permuted-mnist --model mlp --plugin warm_n_iter

> python3 new_src/runner.py --strategy agem --benchmark permuted-mnist --model mlp --plugin agem_n_iter
'''
