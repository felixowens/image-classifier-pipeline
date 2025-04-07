import torch
import torch.nn as nn
from image_classifier_pipeline.lib.logger import setup_logger

# TODO: make this use a dataset config property rather than a hardcoded value
DINO_FEATURE_DIMENSIONS = 768

logger = setup_logger(__name__)


class SimpleClassifierHead(nn.Module):
    """Simple Linear Classifier Head."""

    def __init__(self, input_dim: int = DINO_FEATURE_DIMENSIONS, num_classes: int = 0):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be greater than 0")
        self.fc = nn.Linear(input_dim, num_classes)
        logger.info(
            f"SimpleClassifierHead initialized with {input_dim} input dimensions and {num_classes} classes"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ComplexClassifierHead(nn.Module):
    """Complex Classifier Head."""

    def __init__(self, input_dim: int = DINO_FEATURE_DIMENSIONS, num_classes: int = 0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        logger.info(
            f"ComplexClassifierHead initialized with {input_dim} input dimensions and {num_classes} classes"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
