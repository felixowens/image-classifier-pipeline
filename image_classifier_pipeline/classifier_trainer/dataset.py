import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Tuple

from image_classifier_pipeline.lib.models import DatasetSplit


class FeatureDataset(TorchDataset[Tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for pre-extracted features and labels."""

    def __init__(self, split: DatasetSplit):
        self.items = split.items
        # Quick check for feature dimension consistency (optional)
        if self.items:
            self._feature_dim = len(self.items[0].features)
            assert all(
                len(item.features) == self._feature_dim for item in self.items
            ), "Inconsistent feature dimensions found in dataset split."

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        # Features are already List[float], convert to tensor
        features_tensor = torch.tensor(item.features, dtype=torch.float32)
        # label_id is the integer representation
        label_tensor = torch.tensor(
            item.label_id, dtype=torch.long
        )  # CrossEntropyLoss expects long
        return features_tensor, label_tensor

    @property
    def feature_dim(self) -> int:
        return self._feature_dim if hasattr(self, "_feature_dim") else 0

    @property
    def num_classes(self) -> int:
        # Infer number of classes from max label_id + 1
        if not self.items:
            return 0
        return max(item.label_id for item in self.items) + 1
