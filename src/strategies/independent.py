class IndependentStrategy:
    def __init__(self, base_class, finetune: bool = True, **kwargs):
        self.base_class = base_class
        self.finetune = finetune
        self.base_kwargs = kwargs
        self.last_strategy = self.base_class(**self.base_kwargs)

    def train(self, experience):
        # Initialize new strategy instance
        new_strategy = self.base_class(**self.base_kwargs)
        # If finetuning and a previous model exists, copy its weights
        if self.finetune and self.last_strategy is not None:
            new_strategy.model.load_state_dict(
                self.last_strategy.model.state_dict()
            )
        # Train this strategy on the current experience
        new_strategy.train(experience)
        # Store for next round
        self.last_strategy = new_strategy

    def eval(self, test_stream):
        # Evaluate the last-trained strategy
        return self.last_strategy.eval(test_stream)