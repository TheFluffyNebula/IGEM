from avalanche.models import SimpleMLP
from .base import BaseModel

class MLPModel(BaseModel):
    def __init__(self, num_classes, hidden_size=100, hidden_layers=2):
        super().__init__(num_classes)
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

    def build(self):
        return SimpleMLP(
            num_classes=self.num_classes,
            hidden_size=self.hidden_size,
            hidden_layers=self.hidden_layers
        )
