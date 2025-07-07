import argparse
import random
import numpy as np
import torch
import os
from avalanche.training.plugins import EvaluationPlugin
from avalanche.distributed.distributed_helper import DistributedHelper
import pandas as pd
from avalanche.logging import TextLogger

from transformers import GPT2Tokenizer

from avalanche.evaluation.metrics import (
    accuracy_metrics, 
    timing_metrics,
    loss_metrics,
    )
def init_tokenizer():
    global TOKENIZER
    TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")


def get_tokenizer():
    return TOKENIZER
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
    all_metrics = [
       # accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
       # timing_metrics(minibatch=True, experience=True, stream=True, epoch=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        proj_metric
    ]

    # only the main process logs interactively
    #loggers = [InteractiveLogger] if DistributedHelper.is_distributed and not DistributedHelper.is_main_process else [InteractiveLogger()]
    loggers=[TextLogger(file=open('evaluation.log', 'w'))]
    #loggers = None
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
        print(f"\n--- Training on Experience {exp_id} ---")
        strategy.train(train_experience)
        print(f"\n--- Evaluating after Experience {exp_id} ---")
        res = strategy.eval(test_stream)
        res["experience"] = exp_id
        results.append(res)
    return results
