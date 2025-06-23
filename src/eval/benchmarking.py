import torch
from torch.utils.data.distributed import DistributedSampler
from avalanche.distributed.distributed_helper import DistributedHelper
from .metrics import ProjectionOverheadMetric
from util import (
    get_device,
    get_eval_plugin,
    set_seed,
    save_results,
    train_and_evaluate,
)
from factories import make_benchmark, make_model, make_strategy, make_plugin


class BenchmarkRunner:
    def __init__(self, args):
        self.args = args
        self.dist = False
        self.device = None
        self.strategy = None
        self.train_stream, self.test_stream = None, None

    def setup_distributed(self):
        self.dist = DistributedHelper.init_distributed(
            random_seed=self.args.seed,
            backend="gloo",
            use_cuda=torch.cuda.is_available()
        )
        if self.dist and DistributedHelper.use_cuda:
            # pin each process to its GPU
            self.args.cuda = DistributedHelper.get_device_id()

    def setup_device_and_seed(self):
        set_seed(self.args.seed)
        self.device = get_device(self.args.cuda)
        print(f"[Rank {DistributedHelper.rank}/{DistributedHelper.world_size}] using device {self.device}")


    def prepare_data(self):
        bench = make_benchmark(
            self.args.benchmark,
            n_experiences=self.args.n_experiences,
            seed=self.args.seed
        )
        # keep references
        self.train_stream, self.test_stream = bench.train_stream, bench.test_stream

        if self.dist:
            # inject DistributedSampler into each train loader
            for exp in self.train_stream:
                exp.dataset = exp.dataset  # no‐op, placeholder if you want to Subset()
        # num_classes is same for train & test
        self.num_classes = bench.n_classes

    def build_model_and_strategy(self, eval=True):
        model = make_model(name=self.args.model, num_classes=self.num_classes).to(self.device)
        if self.dist:
           # pin this process to its GPU
           local_id = DistributedHelper.get_device_id()
           torch.cuda.set_device(local_id)
           self.args.cuda = local_id

           # single DDP wrap
           model = torch.nn.parallel.DistributedDataParallel(
               model,
               device_ids=[local_id],
               output_device=local_id,
               find_unused_parameters=False,
           )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum
        )
        proj_metric = ProjectionOverheadMetric()
        _evaluator= lambda: get_eval_plugin(cuda_id=self.args.cuda, num_classes=self.num_classes,proj_metric=proj_metric)        
        plugin = make_plugin(self.args.plugin,
                               patterns_per_exp=self.args.patterns_per_exp,
                               memory_strength=self.args.memory_strength,
                               sgd_iterations=self.args.sgd_iterations,
                               sample_size=self.args.sample_size,
                               coord_iterations=self.args.coord_iterations,
                               projection_iteration=self.args.projection_iteration,
                               proj_metric=proj_metric)
        self.strategy = make_strategy(
            name=self.args.strategy,
            model=model,
            optimizer=optimizer,
            criterion=torch.nn.CrossEntropyLoss(),
            device=self.device,
            patterns_per_exp=self.args.patterns_per_exp,
            memory_strength=self.args.memory_strength,
            train_mb_size=self.args.train_mb_size,
            train_epochs=self.args.train_epochs,
            eval_mb_size=self.args.eval_mb_size,
            memory_size=self.args.memory_size,
            ewc_lambda=self.args.ewc_lambda,
            sample_size = self.args.sample_size,
            eval_every=0,
            plugins=[plugin],
            evaluator = _evaluator
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
            save_results(results, self.args.output_dir, self.args.result_filename)

        if self.dist:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
