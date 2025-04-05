"""
Dataset Construction Component for Image Classification Pipeline.

This module provides functionality for:
- Loading and validating configuration for dataset construction
- Associating images with labels based on metadata or directory structure
- Splitting the dataset into train, validation, and test sets
- Saving the dataset splits and label mappings to disk
"""

from .builder import DatasetBuilder, Dataset, DatasetSplit, ImageItem
from .config import DatasetConfig, LabelSourceType, TaskType, Task

__all__ = [
    "DatasetBuilder",
    "Dataset",
    "DatasetSplit",
    "ImageItem",
    "DatasetConfig",
    "LabelSourceType",
    "TaskType",
    "Task",
]
