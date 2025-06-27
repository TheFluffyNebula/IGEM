import torch
from torch.utils.data.distributed import DistributedSampler
from avalanche.distributed.distributed_helper import DistributedHelper
from avalanche.training.supervised import Naive
from eval import ProjectionOverheadMetric

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

        # Store any additional args as attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def setup_distributed(self):
        self.dist = DistributedHelper.init_distributed(
            random_seed=self.seed,
            backend="gloo",
            use_cuda=torch.cuda.is_available()
        )
        if self.dist and DistributedHelper.use_cuda:
            # pin each process to its GPU
            self.cuda = DistributedHelper.get_device_id()

    def setup_device_and_seed(self):
        set_seed(self.seed)
        self.device = get_device(self.cuda)
        print(f"[Rank {DistributedHelper.rank}/{DistributedHelper.world_size}] using device {self.device}")

    def prepare_data(self):
        bench = make_benchmark(
            name=self.benchmark,
            n_experiences=self.n_experiences,
            seed=self.seed,
        )
        # keep references
        self.train_stream, self.test_stream = bench.train_stream, bench.test_stream

        if self.dist:
            # inject DistributedSampler into each train loader
            for exp in self.train_stream:
                exp.dataset = exp.dataset  # no‐op, placeholder if you want to Subset()
        # num_classes is same for train & test
        self.num_classes = bench.n_classes

    def build_model_and_plugin(self, eval=True):
        model = make_model(name=self.model, num_classes=self.num_classes).to(self.device)
        if self.dist:
           # pin this process to its GPU
           local_id = DistributedHelper.get_device_id()
           torch.cuda.set_device(local_id)
           self.cuda = local_id

           # single DDP wrap
           model = torch.nn.parallel.DistributedDataParallel(
               model,
               device_ids=[local_id],
               output_device=local_id,
               find_unused_parameters=False,
           )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum
        )
        proj_metric = ProjectionOverheadMetric()
        _evaluator = lambda: get_eval_plugin(
            cuda_id=self.cuda,
            num_classes=self.num_classes,
            proj_metric=proj_metric
        )

        plugin = make_plugin(
            plugin_name=self.plugin,
            patterns_per_exp=self.patterns_per_exp,
            memory_strength=self.memory_strength,
            proj_interval=self.proj_interval,
            pgd_iterations = getattr(self, "pgd_iterations", 3),
            adaptive_lr=getattr(self, "adaptive_lr", False),
            warm_start=getattr(self, "warm_start", False),
            lr=self.lr,
            sample_size=getattr(self, "sample_size", 1),
            proj_metric=proj_metric
        )

        self.strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=torch.nn.CrossEntropyLoss(),
            device=self.device,
            train_mb_size=self.train_mb_size,
            train_epochs=self.train_epochs,
            eval_mb_size=self.eval_mb_size,
            evaluator=_evaluator,
            eval_every=0,
            plugins=[plugin]
        )

        if self.dist:
            # patch in DDP‐friendly loaders
            from types import MethodType
            def train_loader(self, exp, **kw):
                sampler = DistributedSampler(
                    exp.dataset,
                    num_replicas=DistributedHelper.world_size,
                    rank=DistributedHelper.rank,
                    shuffle=True
                )
                return torch.utils.data.DataLoader(
                    exp.dataset,
                    batch_size=self.train_mb_size,
                    sampler=sampler,
                    num_workers=kw.get("num_workers", 4),
                    pin_memory=(self.device.type=="cuda"),
                    drop_last=False
                )
            def eval_loader(self, exp, **kw):
                return torch.utils.data.DataLoader(
                    exp.dataset,
                    batch_size=self.eval_mb_size,
                    shuffle=False,
                    num_workers=kw.get("num_workers", 4),
                    pin_memory=(self.device.type=="cuda")
                )
            self.strategy.train_dataloader = MethodType(train_loader, self.strategy)
            self.strategy.eval_dataloader  = MethodType(eval_loader,  self.strategy)

    def run(self):
        try:
            results = train_and_evaluate(
                strategy=self.strategy,
                train_stream=self.train_stream,
                test_stream=self.test_stream
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[Rank {DistributedHelper.rank}] OOM!\n" +
                torch.cuda.memory_summary(abbreviated=True))
            raise
        # only rank 0 writes
        if not self.dist or DistributedHelper.is_main_process:
            save_results(results, self.output_dir, self.result_filename)

        if self.dist:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
