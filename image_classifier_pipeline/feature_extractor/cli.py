import typer

from .dino import DinoFeatureExtractor

app = typer.Typer(help="Dataset Feature Extraction Component")


@app.command()
def extract(
    dataset_path: str = typer.Argument(..., help="Path to the dataset folder"),
    output_dir: str = typer.Argument(..., help="Path to save the output datasets"),
):
    """
    Extract features from images using DINO.
    """
    try:
        extractor = DinoFeatureExtractor(dataset_path)
        extractor.extract_features()
        extractor.save_features(output_dir)

        typer.echo(f"Features successfully extracted and saved to {output_dir}")

    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
