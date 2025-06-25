class BaseModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def build(self):
        """Override to return actual model."""
        raise NotImplementedError
