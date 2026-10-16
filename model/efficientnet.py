from optastra.backbones import Backbone
from optastra.necks import Neck
from optastra.heads import Head

from optastra.core.build import build_sequential_model


def get_model(in_channels=3, num_classes=34, dropout=0.0):
    """
    Returns a sequential model with EfficientNet backbone,
    global average pooling neck, and a vanilla classification head.
    
    EfficientNet -> Global Average Pooling -> Vanilla Classification Head

    Vanilla Classification Head consists of 2 layers of MLP:
    Linear -> ReLU Activation -> Dropout
    """
    model = build_sequential_model(
        backbone=("efficientnet_b0", {"in_channels": in_channels}),
        necks=["global_avg_pool"],
        head=("vanilla_classification_head", {"num_classes": num_classes, "hidden_features": 2048, "dropout": dropout})
    )
    return model
