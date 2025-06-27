from models.mlp_model import MLPModel
from models.resnet_model import ResNetModel

def make_model(name: str, num_classes: int):
    if name == "mlp":
        return MLPModel(num_classes=num_classes)
    elif name == "resnet":
        return ResNetModel(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")
