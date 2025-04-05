from enum import Enum
from pathlib import Path
import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class LabelSourceType(str, Enum):
    """Defines how to extract labels from the data."""

    METADATA_FILE = "metadata_file"
    DIRECTORY_STRUCTURE = "directory_structure"


class TaskType(str, Enum):
    """Defines the type of classification task."""

    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"


class Task(BaseModel):
    """Configuration for a single classification task."""

    name: str = Field(..., description="Name of the task (e.g., 'bmi', 'breast_size')")
    type: TaskType = Field(..., description="Type of the task (categorical or ordinal)")
    classes: List[str] = Field(..., description="List of possible class labels")

    # For ordinal tasks, defines the order of classes from smallest to largest
    # Must be a subset of classes if provided
    ordinal_order: Optional[List[str]] = Field(
        None, description="Order of classes for ordinal tasks"
    )

    @field_validator("ordinal_order")
    @classmethod
    def validate_ordinal_order(
        cls, v: Optional[List[str]], info: ValidationInfo
    ) -> Optional[List[str]]:
        """Validate that ordinal_order contains valid class labels."""
        if v is None:
            if info.data.get("type") == TaskType.ORDINAL:
                raise ValueError("ordinal_order must be provided for ordinal tasks")
            return v

        # Check that all elements in ordinal_order are in classes
        classes = info.data.get("classes", [])
        for item in v:
            if item not in classes:
                raise ValueError(f"'{item}' in ordinal_order not found in classes")

        # Check that ordinal_order contains all elements from classes
        if set(v) != set(classes):
            raise ValueError(
                "ordinal_order must contain exactly the same elements as classes"
            )

        return v


class MetadataConfig(BaseModel):
    """Configuration for metadata-based label extraction."""

    path: str = Field(..., description="Path to the metadata file")
    format: Literal["csv", "json"] = Field(
        ..., description="Format of the metadata file"
    )
    image_col: str = Field(..., description="Column name for image paths/filenames")
    label_cols: Dict[str, str] = Field(
        ..., description="Mapping from task names to column names in metadata"
    )


class DirectoryConfig(BaseModel):
    """Configuration for directory structure-based label extraction."""

    pattern: str = Field(
        ...,
        description="Pattern for extracting labels from directory structure. This must include {task_name} and {label} placeholders.",
        examples=["{task_name}/{label}"],
    )

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: str, info: ValidationInfo) -> str:
        """Validate that pattern contains valid placeholders."""
        if not re.match(r"\{.*\}", v):
            raise ValueError("pattern must contain valid placeholders")
        if "{task_name}" not in v:
            raise ValueError("pattern must contain {task_name} placeholder")
        if "{label}" not in v:
            raise ValueError("pattern must contain {label} placeholder")
        if v.count("{task_name}") > 1:
            raise ValueError("pattern must contain only one {task_name} placeholder")
        if v.count("{label}") > 1:
            raise ValueError("pattern must contain only one {label} placeholder")
        return v

    def image_dirs(
        self, root: Path, task_name: str, labels: List[str]
    ) -> Dict[str, Path]:
        """Path to the directory containing the images for each label for a given task."""

        return {
            label: root / self.pattern.format(task_name=task_name, label=label)
            for label in labels
        }


class DatasetConfig(BaseModel):
    """Main configuration for dataset construction."""

    label_source: LabelSourceType = Field(
        ..., description="How to extract labels from the data"
    )
    tasks: List[Task] = Field(..., description="List of classification tasks")
    metadata: Optional[MetadataConfig] = Field(
        None, description="Configuration for metadata-based label extraction"
    )
    directory: Optional[DirectoryConfig] = Field(
        None, description="Configuration for directory-based label extraction"
    )
    stratify_by: Optional[str] = Field(
        None, description="Task name to stratify by when splitting the dataset"
    )

    @field_validator("metadata")
    @classmethod
    def validate_metadata_config(
        cls, v: Optional[MetadataConfig], info: ValidationInfo
    ) -> Optional[MetadataConfig]:
        """Validate that metadata config is provided when label_source is METADATA_FILE."""
        if info.data.get("label_source") == LabelSourceType.METADATA_FILE and v is None:
            raise ValueError(
                "metadata must be provided when label_source is 'metadata_file'"
            )
        return v

    @field_validator("directory")
    @classmethod
    def validate_directory_config(
        cls, v: Optional[DirectoryConfig], info: ValidationInfo
    ) -> Optional[DirectoryConfig]:
        """Validate that directory config is provided when label_source is DIRECTORY_STRUCTURE."""
        if (
            info.data.get("label_source") == LabelSourceType.DIRECTORY_STRUCTURE
            and v is None
        ):
            raise ValueError(
                "directory must be provided when label_source is 'directory_structure'"
            )
        return v

    @field_validator("stratify_by")
    @classmethod
    def validate_stratify_by(
        cls, v: Optional[str], info: ValidationInfo
    ) -> Optional[str]:
        """Validate that stratify_by refers to an existing task."""
        if v is not None:
            task_names = [task.name for task in info.data.get("tasks", [])]
            if v not in task_names:
                raise ValueError(
                    f"stratify_by task '{v}' not found in configured tasks"
                )
        return v
