from enum import Enum
import json
import os
from typing import Dict, List, Optional
from pydantic import BaseModel

from image_classifier_pipeline.lib import setup_logger

logger = setup_logger(__name__)


class ImageItem(BaseModel):
    """Represents a single image with its label."""

    image_path: str
    task_name: str
    label: str
    label_id: int


class ImageFormat(str, Enum):
    PNG = "*.png"
    JPG = "*.jpg"
    JPEG = "*.jpeg"
    GIF = "*.gif"
    BMP = "*.bmp"
    WEBP = "*.webp"


class DatasetSplit(BaseModel):
    """Represents a dataset split (train, validation, or test)."""

    items: List[ImageItem]


class Dataset(BaseModel):
    """Represents a dataset for a single task with splits."""

    task_name: str
    train: DatasetSplit
    test: DatasetSplit
    validation: Optional[DatasetSplit]

    # Store label mapping for this task
    # For categorical tasks: {class_name: index}
    # For ordinal tasks: {class_name: value}
    label_mapping: Dict[str, int]

    # Load a dataset from a folder
    @classmethod
    def load_from_saved_folder(cls, folder_path: str) -> "Dataset":
        """
        Load a dataset from a folder. The folder has the following structure:
        - task/
            - label_mapping.json
            - train.jsonl
            - test.jsonl
            - validation.jsonl (optional)

        Each jsonl file contains a list of ImageItem objects.
        """

        task_name = os.path.basename(folder_path)

        # Load the label mapping
        with open(os.path.join(folder_path, "label_mapping.json"), "r") as f:
            label_mapping = json.load(f)
            logger.info(
                f"Label mapping loaded from {os.path.join(folder_path, 'label_mapping.json')}"
            )

        # Load the splits
        train_items: List[ImageItem] = []
        with open(os.path.join(folder_path, "train.jsonl"), "r") as f:
            for line in f:
                item_json = json.loads(line)
                image_item = ImageItem.model_validate(item_json)
                train_items.append(image_item)

        logger.info(
            f"Train items loaded from {os.path.join(folder_path, 'train.jsonl')}"
        )

        train_dataset = DatasetSplit(items=train_items)

        test_items: List[ImageItem] = []
        with open(os.path.join(folder_path, "test.jsonl"), "r") as f:
            for line in f:
                item_json = json.loads(line)
                image_item = ImageItem.model_validate(item_json)
                test_items.append(image_item)
        test_dataset = DatasetSplit(items=test_items)
        logger.info(f"Test items loaded from {os.path.join(folder_path, 'test.jsonl')}")

        validation_items: List[ImageItem] = []
        if os.path.exists(os.path.join(folder_path, "validation.jsonl")):
            with open(os.path.join(folder_path, "validation.jsonl"), "r") as f:
                for line in f:
                    item_json = json.loads(line)
                    image_item = ImageItem.model_validate(item_json)
                    validation_items.append(image_item)

            logger.info(
                f"Validation items loaded from {os.path.join(folder_path, 'validation.jsonl')}"
            )
            validation_dataset = DatasetSplit(items=validation_items)
        else:
            validation_dataset = None

        # Create the dataset object
        return cls(
            task_name=task_name,
            train=train_dataset,
            test=test_dataset,
            validation=validation_dataset,
            label_mapping=label_mapping,
        )
