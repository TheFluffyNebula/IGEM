import argparse
import random
import numpy as np
import torch
import os
from avalanche.training.plugins import EvaluationPlugin
from avalanche.distributed.distributed_helper import DistributedHelper
import pandas as pd
from avalanche.logging import TextLogger, CSVLogger

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
    # print("TOKENIZER:", TOKENIZER)
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
       accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
       timing_metrics(minibatch=True, experience=True, stream=True, epoch=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        proj_metric
    ]
    file_logger = CSVLogger(
    )
    # only the main process logs interactively
    #loggers = [InteractiveLogger] if DistributedHelper.is_distributed and not DistributedHelper.is_main_process else [InteractiveLogger()]
    loggers=[TextLogger(file=open('evaluation.log', 'w')), file_logger]
    #loggers = None
    return EvaluationPlugin(
        *all_metrics,
        loggers=loggers,
        strict_checks=False,
    )
import numpy as np
def save_results(R_matrix, results, output_dir: str, result_filename: str):

    np_path = os.path.join(output_dir, result_filename) + ".npy"
    np.save(np_path, R_matrix)
    print(f"\nR_matrix saved to: {np_path}")
    
    csv_path = os.path.join(output_dir, result_filename) + ".csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path)
    print(f"\n Results saved to {csv_path}")
    

def train_and_evaluate(
    strategy,
    train_stream,
    test_stream,
    train_epochs
) -> list[dict[str, float]]:
    T = len(train_stream)
    R = np.zeros((T, T))
    results = []
    for j, train_exp in enumerate(train_stream):
        strategy.train(train_exp, eval_streams=None)

        for i, test_exp in enumerate(test_stream):
            metrics = strategy.eval([test_exp])
            #TODO: KEY DYNAMICALLY CHANGE WITH BENCHMARK
            key = f"Top1_Acc_Exp/eval_phase/test_stream/Exp{i:03d}"
            R[i, j] = metrics[key]
            results.append(metrics)

    print("R matrix:\n", R)
    return R, results
