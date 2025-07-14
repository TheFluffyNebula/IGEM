import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from avalanche.distributed.distributed_helper import DistributedHelper
from avalanche.training.supervised import Naive
from eval import ProjectionOverheadMetric
from avalanche.training.plugins import EvaluationPlugin

from util import (
    get_device,
    get_eval_plugin,
    set_seed,
    save_results,
    train_and_evaluate,
)

from factory import make_benchmark, make_model, make_plugin

class Runner:
    def __init__(self, benchmark, plugin, seed, proj_interval, train_mb_size, eval_mb_size,
            n_experiences, train_epochs, lr, momentum, patterns_per_exp, memory_strength, output_dir, cuda, 
            **kwargs
        ):
        self.benchmark = benchmark
        self.plugin = plugin
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
        self.adaptive_lr = True
        self.warm_start = True
        # Store any additional args as attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

    def setup_device_and_seed(self):
        set_seed(self.seed)
        self.device = get_device(self.cuda)

    def prepare_data(self):
        bench = make_benchmark(
            name=self.benchmark,
            n_experiences=self.n_experiences,
            seed=self.seed,
        )
        self.train_stream, self.test_stream = bench.train_stream, bench.test_stream
        self.num_classes = bench.n_classes  

    def build_model_and_plugin(self, eval=True):
        model = make_model(name=self.model, num_classes=self.num_classes).to(self.device)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=1e-2
        )
        proj_metric = ProjectionOverheadMetric()
        plugin = make_plugin(
            plugin_name=self.plugin,
            patterns_per_exp=self.patterns_per_exp,
            n_experiences=self.n_experiences,
            memory_size=self.memory_size,
            memory_strength=self.memory_strength,
            proj_interval=self.proj_interval,
            pgd_iterations = getattr(self, "pgd_iterations", 3),
            adaptive_lr= self.adaptive_lr,#getattr(self, "addons", False) and "adaptive_lr" in self.addons,
            warm_start= self.warm_start,#getattr(self, "addons", False) and "warm_start" in self.addons,
            lr=self.lr,
            sample_size=getattr(self, "sample_size", 1),
            proj_metric=proj_metric
        )
        
        criterion = torch.nn.CrossEntropyLoss()
        self.strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            train_mb_size=self.train_mb_size,
            train_epochs=self.train_epochs,
            eval_mb_size=self.eval_mb_size,
            evaluator=get_eval_plugin(
            cuda_id=self.cuda,
            num_classes=self.num_classes,
            proj_metric=proj_metric
        ),
            eval_every=1,
            plugins=[plugin]
        )

    def run(self):
        try:
            R_matrix, results = train_and_evaluate(
                strategy=self.strategy,
                train_stream=self.train_stream,
                test_stream=self.test_stream,
                train_epochs=self.train_epochs
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[Rank {DistributedHelper.rank}] OOM!\n" +
                torch.cuda.memory_summary(abbreviated=True))
            raise
        s = ""
        if self.plugin == "igem":
            s += "adap_lr" if  self.adaptive_lr else ""
            s += "ws" if self.warm_start else ""
            if s != "":
                s += "_"
        self.result_filename = f"{self.plugin}_{s}{self.proj_interval}_s{self.seed}"
        
        save_results(R_matrix,results, self.output_dir, self.result_filename)
