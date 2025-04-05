import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from image_classifier_pipeline.lib.guards import assert_list
from image_classifier_pipeline.lib.pandas import pandas


from .config import DatasetConfig, LabelSourceType, TaskType


class ImageItem(BaseModel):
    """Represents a single image with its labels."""

    image_path: str
    labels: Dict[str, str]


class DatasetSplit(BaseModel):
    """Represents a dataset split (train, validation, or test)."""

    items: List[ImageItem]


class Dataset(BaseModel):
    """Represents the full dataset with splits."""

    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit

    # Store label mappings for each task
    # For categorical tasks: {task_name: {class_name: index}}
    # For ordinal tasks: {task_name: {class_name: value}}
    label_mappings: Dict[str, Dict[str, int]]


class DatasetBuilder:
    """Builds a dataset from images and labels based on configuration."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.label_mappings = self._create_label_mappings()

    def _create_label_mappings(self) -> Dict[str, Dict[str, int]]:
        """Create label mappings for each task based on the configuration."""
        mappings: Dict[str, Dict[str, int]] = {}

        for task in self.config.tasks:
            if task.type == TaskType.CATEGORICAL:
                # For categorical tasks, map each class to an index
                mappings[task.name] = {cls: idx for idx, cls in enumerate(task.classes)}
            else:  # ORDINAL
                # For ordinal tasks, use the specified order
                mappings[task.name] = {
                    cls: idx for idx, cls in enumerate(task.ordinal_order or [])
                }

        return mappings

    def _load_from_metadata(self, image_root: Path) -> List[ImageItem]:
        """Load images and labels from a metadata file."""
        if not self.config.metadata:
            raise ValueError("Metadata configuration is missing")

        metadata_path = Path(self.config.metadata.path)

        # Load metadata file
        if self.config.metadata.format == "csv":
            df = pandas.read_csv(metadata_path)
        elif self.config.metadata.format == "json":
            df = pandas.read_json(metadata_path)
        else:
            raise ValueError(
                f"Unsupported metadata format: {self.config.metadata.format}"
            )

        items: List[ImageItem] = []
        for _, row in df.iterrows():
            image_file = row[self.config.metadata.image_col]
            # Assert that image_file is a string
            assert isinstance(image_file, str)

            image_path = image_root / image_file
            # Assert that image_path is a valid path
            assert isinstance(image_path, Path)
            assert image_path.exists()
            assert image_path.is_file()
            assert image_path.suffix.lower() in [".png", ".jpg", ".jpeg"]

            # Extract labels for each task
            labels: Dict[str, str] = {}
            for task in self.config.tasks:
                col_name = self.config.metadata.label_cols.get(task.name)
                if not col_name:
                    raise ValueError(
                        f"Column name for task '{task.name}' not found in metadata configuration"
                    )

                label = row[col_name]
                # Assert that label is a string
                assert isinstance(label, str)

                # Assert that label is in the task classes
                assert label in task.classes

                labels[task.name] = label

            # Only add if we have valid labels for all tasks
            if len(labels) == len(self.config.tasks):
                items.append(ImageItem(image_path=str(image_path), labels=labels))

        return items

    def _load_from_directory(self, image_root: Path) -> List[ImageItem]:
        """Load images and labels from directory structure."""
        if not self.config.directory:
            raise ValueError("Directory configuration is missing")

        items: List[ImageItem] = []

        # Walk through the directory structure
        for root, _, files in os.walk(image_root):
            for file in files:
                if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                image_path = Path(root) / file
                path_parts = str(image_path.relative_to(image_root)).split(os.sep)

                # Extract labels based on directory configuration
                labels: Dict[str, str] = {}
                for task in self.config.tasks:
                    pos = self.config.directory.task_positions.get(task.name)
                    if pos is None or pos >= len(path_parts):
                        continue

                    label = path_parts[pos]
                    if label not in task.classes:
                        continue

                    labels[task.name] = label

                # Only add if we have valid labels for all tasks
                if len(labels) == len(self.config.tasks):
                    items.append(ImageItem(image_path=str(image_path), labels=labels))

        return items

    def build(
        self,
        image_root: Union[str, Path],
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
    ) -> Dataset:
        """
        Build the dataset with train, validation, and test splits.

        Args:
            image_root: Path to the root directory containing images
            train_ratio: Ratio of training data
            validation_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            random_state: Random seed for reproducibility

        Returns:
            A Dataset object with train, validation, and test splits
        """
        # Validate split ratios
        if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-10:
            raise ValueError("Split ratios must sum to 1.0")

        image_root = Path(image_root)

        # Load images and labels based on configuration
        if self.config.label_source == LabelSourceType.METADATA_FILE:
            items = self._load_from_metadata(image_root)
        else:  # DIRECTORY_STRUCTURE
            items = self._load_from_directory(image_root)

        if not items:
            raise ValueError("No valid images found")

        # Split the dataset
        train_items, test_items = self._split_dataset(
            items,
            test_size=test_ratio / (train_ratio + validation_ratio + test_ratio),
            random_state=random_state,
        )

        # Further split train into train and validation
        train_items, val_items = self._split_dataset(
            train_items,
            test_size=validation_ratio / (train_ratio + validation_ratio),
            random_state=random_state,
        )

        # Create dataset object
        dataset = Dataset(
            train=DatasetSplit(items=train_items),
            validation=DatasetSplit(items=val_items),
            test=DatasetSplit(items=test_items),
            label_mappings=self.label_mappings,
        )

        return dataset

    def _split_dataset(
        self, items: List[ImageItem], test_size: float, random_state: int
    ) -> Tuple[List[ImageItem], List[ImageItem]]:
        """
        Split the dataset into two parts.

        Args:
            items: List of ImageItem objects
            test_size: Ratio of the second split
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (first_split, second_split)
        """
        # If stratify_by is specified, use it for stratified splitting
        if self.config.stratify_by:
            stratify_labels = [
                item.labels.get(self.config.stratify_by) for item in items
            ]

            stratify_labels = assert_list(stratify_labels)

            # Split the dataset
            idx_train, idx_test = train_test_split(
                range(len(items)),
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_labels,
            )
        else:
            # Simple random split
            idx_train, idx_test = train_test_split(
                range(len(items)), test_size=test_size, random_state=random_state
            )

        # Create the splits
        first_split = [items[i] for i in idx_train]
        second_split = [items[i] for i in idx_test]

        return first_split, second_split

    def save(self, dataset: Dataset, output_dir: Union[str, Path]) -> None:
        """
        Save the dataset splits and label mappings to disk.

        Args:
            dataset: The dataset to save
            output_dir: Directory to save the dataset to
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each split
        for split_name in ["train", "validation", "test"]:
            split_data = getattr(dataset, split_name)
            split_path = output_dir / f"{split_name}.json"

            with open(split_path, "w") as f:
                json.dump(
                    {"items": [item.dict() for item in split_data.items]}, f, indent=2
                )

        # Save label mappings
        mappings_path = output_dir / "label_mappings.json"
        with open(mappings_path, "w") as f:
            json.dump(dataset.label_mappings, f, indent=2)
