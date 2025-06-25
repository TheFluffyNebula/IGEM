from avalanche.models import SlimResNet18
from .base import BaseModel

class ResNetModel(BaseModel):
    def __init__(self, num_classes):
        super().__init__(num_classes)

    def build(self):
        return SlimResNet18(nclasses=self.num_classes)
