import yaml
import argparse
import itertools
from pathlib import Path

def get_param_grid(experiment):
    # print(experiment['global'].items(), end = '\n\n\n')
    for k, v in experiment['global'].items():
        # print(k, v)
        if isinstance(v, str):
            experiment['global'][k] = [v]
    keys = experiment['global'].keys()
    print(keys, end = '\n\n\n')
    param_grid = itertools.product(*experiment['global'].values())    
    print(*param_grid)
    return keys, param_grid
def main(cfg, debug):
    experiment = cfg['debug'] if debug else cfg['full']
    # print(experiment, end='\n\n\n')
    k, param_grid = get_param_grid(experiment)
    
    #print(*all_params)
    # for current_params in param_grid:
    #     print(current_params)

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