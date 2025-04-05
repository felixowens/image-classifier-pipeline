from pathlib import Path
import pandas as pd


class pandas:
    """
    A wrapper around pandas with type hints.
    """

    @staticmethod
    def read_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path)  # type: ignore

    @staticmethod
    def read_json(path: Path) -> pd.DataFrame:
        return pd.read_json(path)  # type: ignore
