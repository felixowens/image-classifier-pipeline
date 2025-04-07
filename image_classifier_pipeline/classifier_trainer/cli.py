import json
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError

from image_classifier_pipeline.lib.models import Dataset
from image_classifier_pipeline.lib.logger import setup_logger

from .trainer import Trainer
from .config import TrainingConfig

app = typer.Typer(help="Image Classifier Training Component")

logger = setup_logger(__name__)


@app.command()
def train(
    dataset_dir: str = typer.Argument(..., help="Path to the dataset directory"),
    config_file: str = typer.Argument(
        ..., help="Path to the training configuration file (YAML/JSON)"
    ),
    output_dir: str = typer.Option(
        "./tmp/output/models", help="Path to save the trained models"
    ),
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
        except ValidationError as e:
            logger.critical(e, exc_info=True)
            raise typer.Exit(code=1)

        # Load dataset
        dataset = Dataset.load_from_saved_folder(dataset_dir)

        # Initialize trainer and train models
        trainer = Trainer(config, dataset, output_dir=output_dir)
        try:
            typer.echo("Training model...")
            trainer.train()
        except Exception as e:
            logger.critical(e, exc_info=True)
            raise typer.Exit(code=1)

        logger.info("Training completed successfully.")

        logger.info("Evaluating model...")
        test_results = trainer.evaluate()
        logger.info("Test results:")
        # Print test results as pretty JSON
        logger.info(json.dumps(test_results, indent=4))

        logger.info("Evaluation completed successfully.")
    except Exception as e:
        logger.critical(e, exc_info=True)

        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
