import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralLoss(nn.Module):
    """
    Consistent Rank Logits (CORAL) loss for ordinal regression.

    Reference:
    Cao, W., Mirjalili, V., & Raschka, S. (2020).
    Rank consistent ordinal regression for neural networks with application to age estimation.
    Pattern Recognition Letters, 140, 325-331.

    The CORAL method transforms a K-class ordinal problem into K-1 binary classification subproblems
    and uses a consistent rank constraint to ensure ordinal relationship among classes.
    """

    def __init__(self):
        super(CoralLoss, self).__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CORAL loss.

        Args:
            logits: Model outputs of shape (batch_size, num_classes-1), representing K-1 binary classifiers
            targets: Target labels of shape (batch_size,) with integer values from 0 to K-1

        Returns:
            The CORAL loss value
        """
        # Number of binary tasks = num_classes - 1
        # num_binary_tasks = logits.size(1)

        # Convert targets to extended binary targets
        # For a target class k, the binary targets are:
        # - 1 for tasks < k (lower ranks)
        # - 0 for tasks >= k (higher or equal ranks)
        extended_targets = torch.zeros_like(logits)
        for i, target in enumerate(targets):
            # For each target, set binary targets for all tasks
            extended_targets[i, :target] = 1

        # Binary cross entropy with logits for each task
        return F.binary_cross_entropy_with_logits(logits, extended_targets)


class CORALModel(nn.Module):
    """
    Implementation of a neural network model using CORAL for ordinal regression.

    This wrapper adapts a standard classifier to output logits for K-1 binary classifiers
    instead of K class probabilities, as required by the CORAL approach.
    """

    def __init__(self, base_model, num_classes):  # type: ignore
        """
        Initialize the CORAL model.

        Args:
            base_model: The base neural network model (feature extractor)
            num_classes: Number of ordinal classes
        """
        super(CORALModel, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        # Get the device of the base model
        self.device = next(base_model.parameters()).device

        # One fewer output than classes (for K-1 binary tasks)
        self.coral_layers = nn.Linear(base_model.fc.out_features, num_classes - 1)
        # Ensure the new layer is on the same device as the base model
        self.coral_layers.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get the features from the base model (removing the final classification layer)
        features = self.base_model(x)
        # Apply the CORAL layers to get the binary classifiers
        return self.coral_layers(features)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make ordinal predictions (0 to num_classes-1).

        Args:
            x: Input data

        Returns:
            Predicted ordinal class (0 to num_classes-1)
        """
        logits = self.forward(x)
        # Apply sigmoid to get probabilities
        probas = torch.sigmoid(logits)
        # Count the number of binary classifiers that predict 1 (class boundary crossings)
        return torch.sum(probas > 0.5, dim=1)
