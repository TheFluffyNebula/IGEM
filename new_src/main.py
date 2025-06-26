import yaml
import argparse
import itertools
from pathlib import Path

def get_param_grid(experiment):
    for k, v in experiment['global'].items():
        if isinstance(v, str):
            experiment['global'][k] = [v]

    # Extract keys (ordered) and value lists
    keys = list(experiment['global'].keys())
    values = list(experiment['global'].values())

    # Create param grid
    param_grid = list(itertools.product(*values))  # list of tuples

    # Reconnect keys to values as dictionaries
    configs = [dict(zip(keys, combo)) for combo in param_grid]
    # print(configs)
    return configs
def main(cfg, debug):
    experiment = cfg['debug'] if debug else cfg['full']
    # print(experiment, end='\n\n\n')
    all_params = get_param_grid(experiment)
    
    # print(all_params)
    for current_params in all_params:
        print(current_params)

    '''
    runner = Runner(benchmark, strategy, strategy_keywords)
    
    '''

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--debug', action='store_true')
    debug = p.parse_args().debug
    
    path = Path(__file__).parent
    file = path / "config.yaml"
    with file.open('r') as f:
        cfg = yaml.safe_load(f)
        main(cfg, debug)