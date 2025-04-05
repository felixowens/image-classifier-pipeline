import json
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError

from .builder import DatasetBuilder
from .config import DatasetConfig

app = typer.Typer(help="Dataset Construction Component")


@app.command()
def build(
    image_dir: str = typer.Argument(..., help="Path to the root image directory"),
    config_file: str = typer.Argument(
        ..., help="Path to the configuration file (YAML/JSON)"
    ),
    output_dir: str = typer.Argument(..., help="Path to save the output datasets"),
    train_ratio: float = typer.Option(0.7, help="Ratio of training data"),
    validation_ratio: float = typer.Option(0.15, help="Ratio of validation data"),
    test_ratio: float = typer.Option(0.15, help="Ratio of test data"),
    random_state: int = typer.Option(42, help="Random seed for reproducibility"),
):
    """
    Build task-specific datasets by associating images with labels and splitting into train/val/test sets.
    """
    try:
        # Load and validate configuration
        config_path = Path(config_file)
        if (
            config_path.suffix.lower() == ".yaml"
            or config_path.suffix.lower() == ".yml"
        ):
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            with open(config_path, "r") as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        # Parse and validate the configuration
        try:
            config = DatasetConfig.model_validate(config_data)
        except ValidationError as e:
            typer.echo(f"Configuration validation error: {e}")
            raise typer.Exit(code=1)

        # Build the datasets
        builder = DatasetBuilder(config)
        try:
            datasets = builder.build(
                image_root=image_dir,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
                random_state=random_state,
            )
        except Exception as e:
            typer.echo(f"Error building datasets: {e}")
            raise typer.Exit(code=1)

        # Save the datasets
        builder.save(datasets, output_dir)

        typer.echo(f"Datasets successfully built and saved to {output_dir}")
        for task_name, dataset in datasets.items():
            typer.echo(f"\nTask: {task_name}")
            typer.echo(f"  - Train set: {len(dataset.train.items)} images")
            typer.echo(f"  - Validation set: {len(dataset.validation.items)} images")
            typer.echo(f"  - Test set: {len(dataset.test.items)} images")

    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
