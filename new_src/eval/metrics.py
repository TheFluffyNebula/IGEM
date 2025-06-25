from avalanche.evaluation.metric_definitions import GenericPluginMetric

class ProjectionOverheadMetric(GenericPluginMetric):
    """
    Measures wall-clock time spent in the projection step each iteration.
    """
    def __init__(self):
        super().__init__(
            metric=None,
            reset_at='experience',
            emit_at='experience',
            mode='train'
        )
        self.start_time = None
        self.elapsed = 0.0

    def reset(self):
        self.start_time = None
        self.elapsed = 0.0

    def result(self):
        # Return projection overhead for this iteration
        # 0/near-0 if no projection
        return self.elapsed
