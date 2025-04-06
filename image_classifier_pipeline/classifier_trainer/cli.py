import json
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError

from image_classifier_pipeline.lib.models import Dataset

from .trainer import Trainer
from .config import TrainingConfig

app = typer.Typer(help="Image Classifier Training Component")


@app.command()
def train(
    dataset_dir: str = typer.Argument(..., help="Path to the dataset directory"),
    config_file: str = typer.Argument(
        ..., help="Path to the training configuration file (YAML/JSON)"
    ),
    output_dir: str = typer.Option("./models", help="Path to save the trained models"),
    random_state: int = typer.Option(42, help="Random seed for reproducibility"),
):
    """
    Train image classifiers using the specified dataset and configuration.
    """
    try:
        # Load and validate configuration
        config_path = Path(config_file)
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            with open(config_path, "r") as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        # Parse and validate the configuration
        try:
            config = TrainingConfig.model_validate(config_data)
            typer.echo(f"Training configuration: {config}")
        except ValidationError as e:
            typer.echo(f"Configuration validation error: {e}")
            raise typer.Exit(code=1)

        # Load dataset
        dataset = Dataset.load_from_saved_folder(dataset_dir)

        # Initialize trainer and train models
        trainer = Trainer(config, dataset)
        try:
            results = trainer.train()
        except Exception as e:
            typer.echo(f"Error during training: {e}")
            raise typer.Exit(code=1)

        raise NotImplementedError("Not implemented")

        # Save the trained models
        trainer.save_models(results, output_dir)

        typer.echo(f"Models successfully trained and saved to {output_dir}")
        for task_name, result in results.items():
            typer.echo(f"\nTask: {task_name}")
            typer.echo(
                f"  - Validation accuracy: {result.metrics.get('val_accuracy', 'N/A'):.4f}"
            )
            typer.echo(
                f"  - Test accuracy: {result.metrics.get('test_accuracy', 'N/A'):.4f}"
            )

    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
