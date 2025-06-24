import argparse
import random
import numpy as np
import torch
import os
from avalanche.training.plugins import EvaluationPlugin
from avalanche.distributed.distributed_helper import DistributedHelper
import pandas as pd

from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics, 
    loss_metrics, 
    timing_metrics, 
    cpu_usage_metrics,
    gpu_usage_metrics,
    ram_usage_metrics,
    confusion_matrix_metrics, 
    disk_usage_metrics,
    bwt_metrics, 
    forward_transfer_metrics,
    )
from avalanche.logging import InteractiveLogger

def parse_args():
    p = argparse.ArgumentParser(
        description="Generic Avalanche-based script to benchmark multiple CL strategies on MNIST variants."
    )

    # General options
    p.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility (default: 123).",
    )
    p.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Zero-indexed CUDA device (set to -1 to force CPU).",
    )
    p.add_argument(
        "--strategy",
        type=str,
        default="gem",
        choices=["single", "independent", "multimodal", "gem", "agem", "icarl", "ewc"],
        help="Which continual‐learning strategy to run.",
    )
    p.add_argument(
        "--plugin",
        type=str,
        default=None,
        choices=["sgem", "sgem_warm", "sgem_ggt", "sgem_best_learning_rate"],
        help="Which plugin to run for the strategy."
    )
    # Model arguments
    p.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp","resnet18", "multimodal"],
        help="Which underlying model to use."
    )
    # Benchmark/dataset options
    p.add_argument(
        "--benchmark",
        type=str,
        default="permuted-mnist",
        choices=["permuted-mnist", "rotated-mnist", "cifar100"],
        help="Which benchmark to use.",
    )
    p.add_argument(
        "--n_experiences",
        type=int,
        default=5,
        help="Number of experiences (tasks) in the chosen benchmark (default: 5).",
    )

    # Training hyperparameters
    p.add_argument(
        "--train_mb_size",
        type=int,
        default=32,
        help="Mini‐batch size for training (default: 32).",
    )
    p.add_argument(
        "--train_epochs",
        type=int,
        default=2,
        help="Number of epochs per experience (default: 2).",
    )
    p.add_argument(
        "--eval_mb_size",
        type=int,
        default=32,
        help="Mini‐batch size for evaluation (default: 32).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for the SGD optimizer (default: 0.001).",
    )
    p.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for the SGD optimizer (default: 0.9).",
    )

    # Strategy‐specific hyperparameters
    p.add_argument(
        "--patterns_per_exp",
        type=int,
        default=256,
        help="Number of memory patterns per experience (used by GEM/AGEM) (default: 256).",
    )
    p.add_argument(
        "--memory_strength",
        type=float,
        default=0.5,
        help="Memory strength for GEM variants (only used if strategy supports it) (default: 0.5).",
    )

    # Output options
    p.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save the evaluation CSVs (default: ./eval_results).",
    )
    p.add_argument(
        "--result_filename",
        type=str,
        default="results.csv",
        help="Filename for the CSV containing evaluation results (default: results.csv).",
    )

    return p.parse_args()



def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cuda_id: int) -> torch.device:
    if torch.cuda.is_available() and cuda_id >= 0:
        return torch.device(f"cuda:{cuda_id}")
    return torch.device("cpu")


def get_eval_plugin(cuda_id, num_classes: int, proj_metric) -> EvaluationPlugin:
    from eval.metrics import(
        ConstraintViolationCountMetric, ProjectionOverheadMetric
        ) 
    all_metrics = [
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        #loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        #forgetting_metrics(experience=True, stream=True),
        timing_metrics(minibatch=True, experience=True, stream=True, epoch=True),
        #bwt_metrics(experience=True, stream=True), 
        #forward_transfer_metrics(experience=True, stream=True),
        #cpu_usage_metrics(experience=True),
        #gpu_usage_metrics(experience=True, gpu_id=cuda_id),
        #ram_usage_metrics(experience=True),
        #disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        #confusion_matrix_metrics(num_classes=num_classes, save_image=False, stream=True),
        # our new ones:
        #ConstraintViolationCountMetric(), 
        proj_metric
    ]

    # only the main process logs interactively
    #loggers = [InteractiveLogger] if DistributedHelper.is_distributed and not DistributedHelper.is_main_process else [InteractiveLogger()]
    #loggers=[InteractiveLogger()]
    loggers = None
    return EvaluationPlugin(
        *all_metrics,
        loggers=loggers,
        strict_checks=False,
    )

def save_results(results: list[dict[str, float]], output_dir: str, result_filename: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, result_filename)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    

def train_and_evaluate(
    strategy,
    train_stream,
    test_stream,
) -> list[dict[str, float]]:

    results = []
    strategy.eval(test_stream)
    for exp_id, train_experience in enumerate(train_stream, start=1):
        print(f"\n--- Training on Experience {exp_id} (classes: {train_experience.classes_in_this_experience}) ---")
        strategy.train(train_experience)
        print(f"\n--- Evaluating after Experience {exp_id} ---")
        res = strategy.eval(test_stream)
        res["experience"] = exp_id
        results.append(res)
    return results
