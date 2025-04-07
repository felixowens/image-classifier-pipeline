from enum import Enum
from pathlib import Path
import re
from typing import Dict, List, Literal, Optional, Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class LabelSourceType(str, Enum):
    """Defines how to extract labels from the data."""

    METADATA_FILE = "metadata_file"
    DIRECTORY_STRUCTURE = "directory_structure"
    FILENAME_PATTERN = "filename_pattern"


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


class DatasetSplit(BaseModel):
    """
    Configuration for a single dataset split.

    Validate that split_mapping is a valid mapping from task names to split ratios.
        The sum of the values must be 1.
        The keys must be "train", "validation", and "test".
        Train and validation must be non-zero.
    """

    train: float = Field(..., description="Ratio of training data")
    validation: float = Field(..., description="Ratio of validation data")
    test: float = Field(..., description="Ratio of test data")

    # @field_validator("train", "validation", "test")
    # @classmethod
    # def validate_split(cls, v: float, info: ValidationInfo) -> float:
    #     """Validate that the split is between 0 and 1."""
    #     if v < 0 or v > 1:
    #         raise ValueError("split must be between 0 and 1")
    #     return v

    # @field_validator("train", "validation", "test")
    # @classmethod
    # def validate_split_sum(cls, v: float, info: ValidationInfo) -> float:
    #     """Validate that the sum of the splits is 1."""
    #     if sum(info.data.values()) != 1:
    #         raise ValueError("split_mapping must sum to 1")
    #     return v

    # @field_validator("train", "test")
    # @classmethod
    # def validate_split_non_zero(cls, v: float, info: ValidationInfo) -> float:
    #     """Validate that the train and test splits are non-zero."""
    #     if v <= 0:
    #         raise ValueError("train and test splits must be non-zero")
    #     return v


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


class ValueMapping(BaseModel):
    """Configuration for mapping values to classes."""

    type: Literal["range", "exact", "regex"] = Field(
        "range", description="Type of mapping to apply"
    )

    # For range mapping (e.g., age ranges)
    ranges: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="List of range definitions, each with 'min', 'max', and 'label'",
    )

    # For exact value mapping (e.g., gender codes)
    values: Optional[Dict[str, str]] = Field(
        None, description="Mapping from exact values to labels"
    )

    # For regex-based mapping
    patterns: Optional[Dict[str, str]] = Field(
        None, description="Mapping from regex patterns to labels"
    )

    @field_validator("ranges")
    @classmethod
    def validate_ranges(
        cls, v: Optional[List[Dict[str, Any]]], info: ValidationInfo
    ) -> Optional[List[Dict[str, Any]]]:
        """Validate that ranges are properly defined."""
        if v is None:
            return v

        if info.data.get("type") != "range":
            raise ValueError("ranges should only be provided when type is 'range'")

        for i, range_def in enumerate(v):
            if "min" not in range_def and "max" not in range_def:
                raise ValueError(
                    f"Range definition at index {i} must have at least one of 'min' or 'max'"
                )
            if "label" not in range_def:
                raise ValueError(f"Range definition at index {i} must have a 'label'")

        return v

    @field_validator("values")
    @classmethod
    def validate_values(
        cls, v: Optional[Dict[str, str]], info: ValidationInfo
    ) -> Optional[Dict[str, str]]:
        """Validate that values mapping is provided when type is 'exact'."""
        if v is None and info.data.get("type") == "exact":
            raise ValueError("values must be provided when type is 'exact'")
        return v

    @field_validator("patterns")
    @classmethod
    def validate_patterns(
        cls, v: Optional[Dict[str, str]], info: ValidationInfo
    ) -> Optional[Dict[str, str]]:
        """Validate that patterns mapping is provided when type is 'regex'."""
        if v is None and info.data.get("type") == "regex":
            raise ValueError("patterns must be provided when type is 'regex'")
        return v

    def map_value(self, value: str) -> str:
        """Map a value to a label based on the mapping configuration."""
        if self.type == "range":
            try:
                # Convert to float for range comparison
                num_value = float(value)
                for range_def in self.ranges or []:
                    min_val = range_def.get("min", float("-inf"))
                    max_val = range_def.get("max", float("inf"))
                    if min_val <= num_value <= max_val:
                        return range_def["label"]
                # If no range matches, return the original value
                return value
            except ValueError:
                # If value can't be converted to float, return as is
                return value
        elif self.type == "exact":
            # Direct mapping from value to label
            return self.values.get(value, value) if self.values else value
        elif self.type == "regex":
            # Match against regex patterns
            if self.patterns:
                for pattern, label in self.patterns.items():
                    if re.match(pattern, value):
                        return label
            return value
        else:
            return value


class FilenameConfig(BaseModel):
    """Configuration for filename-based label extraction."""

    pattern: str = Field(
        ...,
        description="Regex pattern for extracting fields from filenames. Use named capture groups.",
        examples=[r"(?P<age>\d+)_(?P<gender>[mf])_(?P<race>\w+)_(?P<datetime>.+)\.jpg"],
    )

    task_mappings: Dict[str, str] = Field(
        ...,
        description="Mapping from task names to filename fields",
        examples=[{"age_category": "age", "gender": "gender"}],
    )

    value_mappings: Dict[str, ValueMapping] = Field(
        default_factory=dict,
        description="Optional mappings to transform extracted values to labels",
    )

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: str, info: ValidationInfo) -> str:
        """Validate that pattern is a valid regex with named capture groups."""
        try:
            regex = re.compile(v)
            if not regex.groupindex:
                raise ValueError(
                    "pattern must contain at least one named capture group"
                )
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
        return v

    @field_validator("task_mappings")
    @classmethod
    def validate_task_mappings(
        cls, v: Dict[str, str], info: ValidationInfo
    ) -> Dict[str, str]:
        """Validate that task_mappings references valid capture groups in the pattern."""
        if not v:
            raise ValueError("task_mappings must not be empty")

        try:
            pattern = info.data.get("pattern", "")
            regex = re.compile(pattern)
            capture_groups = set(regex.groupindex.keys())

            for task, field in v.items():
                if field not in capture_groups:
                    raise ValueError(
                        f"Field '{field}' for task '{task}' not found in pattern capture groups"
                    )
        except re.error:
            # Skip this validation if the pattern is invalid (it will be caught by the pattern validator)
            pass

        return v


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
    filename: Optional[FilenameConfig] = Field(
        None, description="Configuration for filename-based label extraction"
    )
    stratify_by: Optional[str] = Field(
        None, description="Task name to stratify by when splitting the dataset"
    )

    split_mapping: DatasetSplit = Field(
        default=DatasetSplit(train=0.8, validation=0, test=0.2),
        description="Mapping of dataset splits.",
    )

    limit: Optional[int] = Field(
        None, description="Limit the number of files to process"
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

    @field_validator("filename")
    @classmethod
    def validate_filename_config(
        cls, v: Optional[FilenameConfig], info: ValidationInfo
    ) -> Optional[FilenameConfig]:
        """Validate that filename config is provided when label_source is FILENAME_PATTERN."""
        if (
            info.data.get("label_source") == LabelSourceType.FILENAME_PATTERN
            and v is None
        ):
            raise ValueError(
                "filename must be provided when label_source is 'filename_pattern'"
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
