import time
from avalanche.evaluation.metric_definitions import GenericPluginMetric


class ConstraintViolationCountMetric(GenericPluginMetric):
    """
    Counts how many gradient constraints are violated at each training iteration.
    """
    def __init__(self):
        super().__init__(
            metric=None,
            reset_at='experience',  # reset at start of each iteration
            emit_at='experience',   # emit after each iteration
            mode='train'           # only during training
        )
        self.current_value = 0

    def reset(self):
        # Reset violation count for new iteration
        self.current_value = 0

    def after_backward(self, strategy):
        # Count violations: gradient · reference_grad < 0
        violations = 0
        for p in strategy.model.parameters():
            if p.grad is None:
                continue
            ref = getattr(p, "ref_grad_", None)
            if ref is None:
                continue
            # dot product negative => violated constraint
            if (p.grad.view(-1) * ref.view(-1)).sum().item() < 0:
                violations += 1
        self.current_value = violations

    def result(self):
        # Return count for this iteration
        return self.current_value


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
