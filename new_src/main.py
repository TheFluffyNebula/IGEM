import yaml
import argparse
import itertools
from pathlib import Path
from runner import Runner

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def get_param_grid(experiment):
    keys = list(experiment['global'].keys())
    values = list(experiment['global'].values())
    param_grid = list(itertools.product(*values)) 
    configs = [dict(zip(keys, combo)) for combo in param_grid]
    return configs

def main(cfg, debug):
    experiment = cfg['debug'] if debug else cfg['full']
    global_configs = get_param_grid(experiment)
    for plugin, plugin_settings in experiment["plugins"].items():
        plugin_keys = list(plugin_settings.keys())
        plugin_values = list(plugin_settings.values())
        plugin_grid = list(itertools.product(*plugin_values))

        for plugin_config in plugin_grid:
            strat_kwargs = dict(zip(plugin_keys, plugin_config))
            
            for benchmark, models in experiment["benchmarks"].items():
                for model in models:
                    for global_params in global_configs:
                        out_dir = global_params["output_dir"] + benchmark + "/"
                        config = {
                            **global_params,
                            "output_dir" : out_dir,
                            "benchmark": benchmark,
                            "plugin": plugin,
                            "model": model,
                            **strat_kwargs  
                        }
                        
                        print(config['benchmark'], config['plugin'], config['model'], config)
                        runner = Runner(**config)
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