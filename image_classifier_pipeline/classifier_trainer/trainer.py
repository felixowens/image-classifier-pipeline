import json
import os
from typing import Any, Dict, Optional, Tuple, cast

import aim
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from image_classifier_pipeline.classifier_trainer.config import TrainingConfig
from image_classifier_pipeline.classifier_trainer.dataset import FeatureDataset
from image_classifier_pipeline.classifier_trainer.model import SimpleClassifierHead
from image_classifier_pipeline.lib.models import Dataset
from image_classifier_pipeline.lib.logger import setup_logger

logger = setup_logger(__name__)

criterion_standard = torch.nn.CrossEntropyLoss()


class Trainer:
    """
    Trainer for a classifier.
    """

    def __init__(
        self,
        config: TrainingConfig,
        dataset: Dataset,
        output_dir: str = "training_output",
    ):
        self.config = config
        self.dataset = dataset

        self.output_dir = os.path.join(
            output_dir,
            self.config.model_information.name,
            self.config.model_information.version,
        )
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Training output will be saved to: {self.output_dir}")

        logger.info(
            f"Dataset '{dataset.task_name}' loaded with {len(dataset.train.items)} training items."
        )
        if dataset.validation:
            logger.info(f"{len(dataset.validation.items)} validation items.")
        logger.info(f"{len(dataset.test.items)} test items.")

        logger.info(f"Label Mapping: {dataset.label_mapping}")
        self.num_classes = len(dataset.label_mapping)
        self.label_names = sorted(
            dataset.label_mapping, key=lambda x: dataset.label_mapping[x]
        )

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Set seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # --- Datasets and Dataloaders ---
        self.train_dataset = FeatureDataset(dataset.train)
        self.test_dataset = FeatureDataset(dataset.test)
        self.val_dataset = (
            FeatureDataset(dataset.validation) if dataset.validation else None
        )

        # Ensure consistent feature dims and num_classes detection
        self.feature_dim = self.train_dataset.feature_dim
        assert (
            self.feature_dim > 0
        ), "Could not determine feature dimension from training data."
        assert (
            self.test_dataset.feature_dim == self.feature_dim
        ), "Test feature dimension mismatch."
        if self.val_dataset:
            assert (
                self.val_dataset.feature_dim == self.feature_dim
            ), "Validation feature dimension mismatch."
            assert (
                self.val_dataset.num_classes == self.num_classes
            ), "Validation class count mismatch."
        assert (
            self.train_dataset.num_classes == self.num_classes
        ), "Train class count mismatch."
        assert (
            self.test_dataset.num_classes == self.num_classes
        ), "Test class count mismatch."

        logger.info(f"Detected Feature Dimension: {self.feature_dim}")
        logger.info(f"Detected Number of Classes: {self.num_classes}")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.hyperparameters.batch_size,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.hyperparameters.batch_size,
            shuffle=False,
        )
        self.val_loader = (
            DataLoader(
                self.val_dataset,
                batch_size=config.hyperparameters.batch_size,
                shuffle=False,
            )
            if self.val_dataset
            else None
        )

        # --- Model ---
        self.model = SimpleClassifierHead(
            input_dim=self.feature_dim, num_classes=self.num_classes
        )
        self.model.to(self.device)

        # --- Loss Function (Choose one) ---
        # self.criterion = criterion_ordinal # Use this if you have a proper ordinal loss
        self.criterion = criterion_standard
        self.criterion.to(self.device)
        # Indicate if using standard loss for ordinal-like task
        logger.warning(
            "Using standard CrossEntropyLoss. Consider an Ordinal Loss (e.g., CORAL) for better results if classes are ordered."
        )

        # --- Optimizer ---
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.hyperparameters.learning_rate
        )
        # Optional: Add LR scheduler here if needed
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)

        # --- Aim Tracking ---
        self.aim_run = aim.Run(
            experiment=f"{config.model_information.name}_v{config.model_information.version}"
        )
        # Sanitize config for Aim (it might not like Pydantic models directly)
        config_json_string = config.model_dump_json()
        config_dict = json.loads(config_json_string)
        self.aim_run["hparams"] = config_dict

        logger.info(
            f"Aim run initialized. Check UI or logs at: {self.aim_run.repo.path}"
        )

        # --- Tracking Best Model ---
        self.best_val_metric = float("inf")  # Lower is better for loss or MAE
        # self.best_val_metric = float('-inf') # Higher is better for accuracy
        self.best_epoch = -1

    def _calculate_metrics(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        """Calculates loss, accuracy, and MAE (for ordinal interpretation)."""
        loss = self.criterion(logits, labels).item()

        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean().item()

        # Mean Absolute Error (MAE) on class indices - useful for ordinal tasks
        # Assumes labels are 0, 1, 2... corresponding to order
        mae = torch.abs(preds.float() - labels.float()).mean().item()

        return {"loss": loss, "accuracy": accuracy, "mae": mae}

    def _run_epoch(
        self,
        loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
        is_training: bool = True,
    ) -> Dict[str, float]:
        """Runs a single epoch of training or validation."""
        if is_training:
            self.model.train()
            context = torch.enable_grad()
            aim_context = {"subset": "train"}
        else:
            self.model.eval()
            context = torch.no_grad()
            aim_context = {
                "subset": "val" if self.val_loader else "test"
            }  # Label context appropriately

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_mae = 0.0
        num_batches = len(loader)

        logger.debug(
            f"Running {'Train' if is_training else 'Eval'} on {num_batches} batches. Aim context: {aim_context}"
        )

        pbar = tqdm(loader, desc=f"{'Train' if is_training else 'Eval'}", leave=False)
        with context:
            for _, (features, labels) in enumerate(pbar):
                features, labels = features.to(self.device), labels.to(self.device)

                if is_training:
                    self.optimizer.zero_grad()

                # Forward pass
                logits = self.model(features)
                loss = self.criterion(logits, labels)

                if is_training:
                    loss.backward()
                    self.optimizer.step()

                # Calculate batch metrics
                batch_metrics = self._calculate_metrics(logits, labels)
                epoch_loss += batch_metrics["loss"]
                epoch_accuracy += batch_metrics["accuracy"]
                epoch_mae += batch_metrics["mae"]

                # Log batch metrics to Aim (optional, can be verbose)
                # self.aim_run.track(batch_metrics["loss"], name='batch_loss', context=aim_context)
                # self.aim_run.track(batch_metrics["accuracy"], name='batch_accuracy', context=aim_context)
                # self.aim_run.track(batch_metrics["mae"], name='batch_mae', context=aim_context)

                pbar.set_postfix(
                    {
                        "loss": f"{batch_metrics['loss']:.4f}",
                        "acc": f"{batch_metrics['accuracy']:.4f}",
                        "mae": f"{batch_metrics['mae']:.4f}",
                    }
                )

        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        avg_mae = epoch_mae / num_batches

        return {"loss": avg_loss, "accuracy": avg_accuracy, "mae": avg_mae}

    def train(self) -> None:
        """Runs the main training loop."""
        logger.info("Starting training...")
        total_epochs = self.config.hyperparameters.num_epochs

        for epoch in range(total_epochs):
            logger.info(f"\n--- Epoch {epoch+1}/{total_epochs} ---")

            # --- Training Phase ---
            train_metrics = self._run_epoch(self.train_loader, is_training=True)
            logger.info(
                f"Train | Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, MAE: {train_metrics['mae']:.4f}"
            )
            self.aim_run.track(
                train_metrics["loss"],
                name="epoch_loss",
                epoch=epoch,
                context={"subset": "train"},
            )
            self.aim_run.track(
                train_metrics["accuracy"],
                name="epoch_accuracy",
                epoch=epoch,
                context={"subset": "train"},
            )
            self.aim_run.track(
                train_metrics["mae"],
                name="epoch_mae",
                epoch=epoch,
                context={"subset": "train"},
            )

            # --- Validation Phase ---
            val_metrics = None
            if self.val_loader:
                val_metrics = self._run_epoch(self.val_loader, is_training=False)
                logger.info(
                    f"Val   | Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, MAE: {val_metrics['mae']:.4f}"
                )
                self.aim_run.track(
                    val_metrics["loss"],
                    name="epoch_loss",
                    epoch=epoch,
                    context={"subset": "val"},
                )
                self.aim_run.track(
                    val_metrics["accuracy"],
                    name="epoch_accuracy",
                    epoch=epoch,
                    context={"subset": "val"},
                )
                self.aim_run.track(
                    val_metrics["mae"],
                    name="epoch_mae",
                    epoch=epoch,
                    context={"subset": "val"},
                )

                # --- Checkpointing ---
                # Using MAE as the metric to optimize for ordinal tasks (lower is better)
                current_val_metric = val_metrics["mae"]
                # Or use accuracy if that's preferred:
                # current_val_metric = -val_metrics['accuracy'] # Negate accuracy so lower is better

                if current_val_metric < self.best_val_metric:
                    self.best_val_metric = current_val_metric
                    self.best_epoch = epoch
                    logger.info(
                        f"✨ New best validation metric ({self.best_val_metric:.4f}) at epoch {epoch+1}. Saving model..."
                    )
                    self.save("best_model.pth")
            # Optional: Adjust learning rate with scheduler
            # if self.scheduler and val_metrics:
            #    self.scheduler.step(val_metrics['loss']) # Or other metric

        logger.info("Training finished.")
        logger.info(
            f"Best validation metric ({self.best_val_metric:.4f}) achieved at epoch {self.best_epoch+1}"
        )
        # Save the final model as well
        self.save("final_model.pth")
        # Close the Aim run
        self.aim_run.close()

    def evaluate(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluates the model on the test set."""
        if model_path:
            logger.info(f"Loading model from {model_path} for evaluation.")
            try:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
                return {}
        else:
            logger.warning(
                "No model path provided for evaluation, using current model state."
            )
            # Attempt to load the best model if available
            best_model_path = os.path.join(self.output_dir, "best_model.pth")
            if os.path.exists(best_model_path):
                logger.info(f"Attempting to load best model from {best_model_path}")
                self.model.load_state_dict(
                    torch.load(best_model_path, map_location=self.device)
                )
            else:
                logger.warning(
                    "Best model checkpoint not found. Evaluating with potentially untrained/final model."
                )

        logger.info("Evaluating on the test set...")
        self.model.eval()
        all_preds = []
        all_labels = []
        all_logits = []  # Collect logits if needed for calibration etc.

        with torch.no_grad():
            for features, labels in tqdm(self.test_loader, desc="Testing"):
                features, labels = features.to(self.device), labels.to(self.device)
                logits = self.model(features)

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())  # Optional

        # Calculate final metrics
        test_loss = self.criterion(
            torch.tensor(all_logits).to(self.device),
            torch.tensor(all_labels).to(self.device),
        ).item()  # Recompute loss on all test data
        test_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        test_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))

        logger.info(f"\n--- Test Set Evaluation ---")
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test MAE: {test_mae:.4f}")

        # --- Detailed Report & Confusion Matrix ---
        report = cast(
            Dict[str, Dict[str, float]],
            classification_report(
                all_labels,
                all_preds,
                target_names=self.label_names,
                output_dict=True,
                zero_division=0,
            ),
        )

        cm = confusion_matrix(all_labels, all_preds)
        cm_df = pd.DataFrame(cm, index=self.label_names, columns=self.label_names)

        # Log to Aim
        eval_aim_run = aim.Run(
            experiment=f"{self.config.model_information.name}_v{self.config.model_information.version}_EVAL"
        )
        eval_aim_run["hparams"] = self.config.model_dump_json()
        eval_aim_run.track(test_loss, name="test_loss")
        eval_aim_run.track(test_accuracy, name="test_accuracy")
        eval_aim_run.track(test_mae, name="test_mae")

        # Track report?

        # Create confusion matrix figure
        try:
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            cm_path = os.path.join(self.output_dir, "confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close()
            # Log image to Aim
            eval_aim_run.track(aim.Image(cm_path), name="confusion_matrix")
            logger.info(f"Confusion matrix saved to {cm_path}")
        except Exception as e:
            logger.error(f"Could not generate or save confusion matrix plot: {e}")

        eval_aim_run.close()

        return {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_mae": test_mae,
            "classification_report": report,
            "confusion_matrix": cm_df.to_dict(),  # Return as dict
        }

    def save(self, filename: str = "model.pth") -> None:
        """Saves the model state dictionary."""
        save_path = os.path.join(self.output_dir, filename)
        try:
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model to {save_path}: {e}")
