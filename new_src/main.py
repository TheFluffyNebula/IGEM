import yaml
import argparse
import itertools
from pathlib import Path
from runner import Runner

def get_param_grid(experiment):
    # forgot to wrap output directory in a list
    # for k, v in experiment['global'].items():
    #     if isinstance(v, str):
    #         experiment['global'][k] = [v]

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
    global_configs = get_param_grid(experiment)
    # for current_params in global_params:
    #     print(current_params)

    # Loop through strategies
    for plugin, plugin_settings in experiment["plugins"].items():
        plugin_keys = list(plugin_settings.keys())
        plugin_values = list(plugin_settings.values())
        plugin_grid = list(itertools.product(*plugin_values))

        for plugin_config in plugin_grid:
            strat_kwargs = dict(zip(plugin_keys, plugin_config))
            
            # Loop through benchmarks + associated models
            for benchmark, models in experiment["benchmarks"].items():
                for model in models:
                    for global_params in global_configs:
                        # Combine everything
                        config = {
                            **global_params,
                            "benchmark": benchmark,
                            "plugin": plugin,
                            "model": model,
                            **strat_kwargs  # e.g., memory_sizes=5120
                        }
                        
                        # print(config['benchmark'], config['plugin'], config['model'])
                        runner = Runner(**config)
                        # if runner.plugin == 'igem':
                        #     print(runner.addons)
                        runner.setup_distributed()
                        runner.setup_device_and_seed()
                        runner.prepare_data()
                        runner.build_model_and_plugin()
                        runner.run()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--debug', action='store_true')
    debug = p.parse_args().debug
    
    path = Path(__file__).parent
    file = path / "config.yaml"
    with file.open('r') as f:
        cfg = yaml.safe_load(f)
        main(cfg, debug)