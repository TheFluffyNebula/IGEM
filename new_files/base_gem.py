from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

class BaseGEMPlugin(SupervisedPlugin):

    def __init__(self, memory_strength: float, proj_interval: int):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory. (basically the number of samples per task)
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        """

        super().__init__()
        self.memory_strength = memory_strength
        self.proj_interval = proj_interval
