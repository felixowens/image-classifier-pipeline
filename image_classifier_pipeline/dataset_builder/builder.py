import json
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

from sklearn.model_selection import train_test_split
from PIL import Image
from image_classifier_pipeline.feature_extractor.dino import DinoFeatureExtractor
from image_classifier_pipeline.lib import (
    setup_logger,
    ImageItem,
    ImageFormat,
    DatasetSplit,
    Dataset,
)

from .config import DatasetConfig, DirectoryConfig, LabelSourceType, TaskType

logger = setup_logger(__name__)


class DatasetBuilder:
    """Builds a dataset from images and labels based on configuration."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.label_mappings = self._create_label_mappings()
        self.feature_extractor = DinoFeatureExtractor()

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

        raise NotImplementedError("Metadata loading not implemented")

    def _load_from_directory(
        self, directory: DirectoryConfig, image_root: Path
    ) -> List[ImageItem]:
        """Load images and labels from directory structure."""

        items: List[ImageItem] = []

        for task in self.config.tasks:
            logger.info(
                f"Loading images from {directory.image_dirs(image_root, task.name, task.classes)}"
            )
            image_dirs = directory.image_dirs(image_root, task.name, task.classes)
            if len(image_dirs) == 0:
                raise ValueError(
                    f"No images found for task {task.name} in {image_root}"
                )

            for label, image_dir in image_dirs.items():
                if not image_dir.exists():
                    raise ValueError(
                        f"Directory {image_dir} does not exist for task {task.name}"
                    )
                if not image_dir.is_dir():
                    raise ValueError(
                        f"Directory {image_dir} is not a directory for task {task.name}"
                    )

                image_files: List[Path] = []
                for format in ImageFormat:
                    image_files.extend(
                        image_dir.glob(format.value, case_sensitive=False)
                    )
                if len(image_files) == 0:
                    raise ValueError(
                        f"No images found for task {task.name} in {image_dir}"
                    )

                # TODO: add progress bar
                for image_path in image_files:
                    image = Image.open(image_path)
                    features = self.feature_extractor.extract_features(image)
                    logger.info(f"Extracted features: {features.shape}")

                    features_list: List[float] = features.flatten().tolist()

                    items.append(
                        ImageItem(
                            image_path=str(image_path),
                            label=label,
                            task_name=task.name,
                            label_id=self.label_mappings[task.name][label],
                            features=features_list,
                        )
                    )

        logger.info(f"Loaded {len(items)} images for {len(self.config.tasks)} tasks")

        return items

    def build(
        self,
        image_root: Union[str, Path],
        random_state: int = 42,
    ) -> Dict[str, Dataset]:
        """
        Build separate datasets for each task with train, validation, and test splits.

        Args:
            image_root: Path to the root directory containing images
            random_state: Random seed for reproducibility

        Returns:
            A dictionary mapping task names to Dataset objects
        """
        logger.info("Starting to build datasets.")

        image_root = Path(image_root)

        # Load all images and labels based on configuration
        if self.config.label_source == LabelSourceType.METADATA_FILE:
            all_items = self._load_from_metadata(image_root)
        else:  # DIRECTORY_STRUCTURE
            assert self.config.directory
            all_items = self._load_from_directory(self.config.directory, image_root)

        if not all_items:
            raise ValueError(f"No valid images found in {image_root}")

        # Create a dataset for each task
        datasets = {}
        for task in self.config.tasks:
            task_name = task.name

            # TODO: process using a reducer so we don't need to filter multiple times

            task_items = [item for item in all_items if item.task_name == task_name]
            logger.info(f"Filtered {len(task_items)} items for task {task_name}")
            if not task_items:
                continue  # Skip tasks with no valid items

            # Calculate the actual split sizes
            split_config = self.config.split_mapping

            # First, split off the test set
            train_val_items, test_items = self._split_dataset(
                task_items,
                test_size=split_config.test,
                random_state=random_state,
                stratify_by=task_name if self.config.stratify_by == task_name else None,
            )

            # Then, if validation is needed, split the remaining data into train and validation
            if split_config.validation > 0:
                # Calculate the validation ratio relative to the train+validation portion
                # validation_ratio = validation / (train + validation)
                validation_ratio = split_config.validation / (
                    split_config.train + split_config.validation
                )

                train_items, val_items = self._split_dataset(
                    train_val_items,
                    test_size=validation_ratio,
                    random_state=random_state,
                    stratify_by=(
                        task_name if self.config.stratify_by == task_name else None
                    ),
                )
            else:
                train_items = train_val_items
                val_items = None

            # Create dataset object for this task
            datasets[task_name] = Dataset(
                task_name=task_name,
                train=DatasetSplit(items=train_items),
                validation=DatasetSplit(items=val_items) if val_items else None,
                test=DatasetSplit(items=test_items),
                label_mapping=self.label_mappings[task_name],
            )

        if not datasets:
            raise ValueError("No valid datasets could be created for any task")

        logger.info("Datasets built successfully.")
        return datasets

    def _split_dataset(
        self,
        items: List[ImageItem],
        test_size: float,
        random_state: int,
        stratify_by: Optional[str] = None,
    ) -> Tuple[List[ImageItem], List[ImageItem]]:
        """
        Split the dataset into two parts.

        Args:
            items: List of ImageItem objects
            test_size: Ratio of the second split
            random_state: Random seed for reproducibility
            stratify_by: Task name to stratify by (overrides config.stratify_by)

        Returns:
            Tuple of (first_split, second_split)
        """
        # Determine which task to stratify by
        stratify_task = stratify_by or self.config.stratify_by

        # If stratify_by is specified, use it for stratified splitting
        if stratify_task:
            stratify_labels = [item.label for item in items]

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

    def save(self, datasets: Dict[str, Dataset], output_dir: Union[str, Path]) -> None:
        """
        Save multiple task-specific datasets to disk in JSONL format.

        Args:
            datasets: Dictionary mapping task names to Dataset objects
            output_dir: Directory to save the datasets to
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save a summary of all tasks
        tasks_summary = {
            task_name: {
                "train_size": len(dataset.train.items),
                "validation_size": (
                    len(dataset.validation.items) if dataset.validation else 0
                ),
                "test_size": len(dataset.test.items),
                "classes": list(dataset.label_mapping.keys()),
            }
            for task_name, dataset in datasets.items()
        }

        with open(output_dir / "tasks_summary.json", "w") as f:
            json.dump(tasks_summary, f, indent=2)

        # Save each task's dataset in its own directory
        for task_name, dataset in datasets.items():
            task_dir = output_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)

            # Save each split
            for split_name in ["train", "validation", "test"]:
                split_data = getattr(dataset, split_name)

                # Skip if the split doesn't exist (e.g., validation might be None)
                if split_data is None:
                    continue

                # Create JSONL file (each line is a valid JSON object)
                split_path = task_dir / f"{split_name}.jsonl"

                with open(split_path, "w") as f:
                    for item in split_data.items:
                        f.write(json.dumps(item.dict()) + "\n")

            # Save label mapping
            mapping_path = task_dir / "label_mapping.json"
            with open(mapping_path, "w") as f:
                json.dump(dataset.label_mapping, f, indent=2)
