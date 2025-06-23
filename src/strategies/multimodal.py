class MultimodalStrategy:
    def __init__(self, base_class, task_count, **kwargs):
        self.base_class = base_class
        self.kwargs = kwargs
        self.task_count = task_count
    
    def train(self, experience):        
        self.last_strategy = self.base_class(**self.kwargs)
        self.last_strategy.train(experience)
    
    def eval(self, test_stream):
        return self.last_strategy.eval(test_stream)
