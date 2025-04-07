from enum import Enum
import re
from pydantic import BaseModel, Field, field_validator


class Task(str, Enum):
    """The intended task of the model."""

    IMAGE_CLASSIFICATION = "image_classification"


class ClassBalanceStrategy(str, Enum):
    """Strategy for handling class imbalance."""

    NONE = "none"  # Use the original dataset as is
    TRUNCATE = "truncate"  # Truncate larger classes to match the smallest class


DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_EPOCHS = 10
DEFAULT_NUM_WORKERS = 4


class ModelInformation(BaseModel):
    """Information about the model to train."""

    name: str = Field(..., description="Name of the model")
    description: str = Field(..., description="Description of the model")
    version: str = Field(..., description="Version of the model")
    task: Task = Field(..., description="The intended task of the model")

    @field_validator("version")
    def validate_version(cls, v: str) -> str:
        """Validate the version of the model."""
        if not re.match(r"^\d+\.\d+\.\d+$", v):
            raise ValueError("Version must be in the semver format x.x.x")
        return v


class Hyperparameters(BaseModel):
    """Hyperparameters for the training process."""

    learning_rate: float = Field(
        DEFAULT_LEARNING_RATE,
        description="Learning rate for the model",
        ge=0,
    )
    batch_size: int = Field(
        DEFAULT_BATCH_SIZE, description="Batch size for training", ge=1
    )
    num_epochs: int = Field(
        DEFAULT_NUM_EPOCHS, description="Number of epochs to train", ge=1
    )


class TrainingConfig(BaseModel):
    """Configuration for training a classifier."""

    model_information: ModelInformation = Field(
        ..., description="Information about the model to train"
    )
    hyperparameters: Hyperparameters = Field(
        ..., description="Hyperparameters for the training process"
    )
    seed: int = Field(..., description="Random seed for reproducibility")
    class_balance_strategy: ClassBalanceStrategy = Field(
        ClassBalanceStrategy.NONE, description="Strategy for handling class imbalance"
    )
