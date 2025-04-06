from image_classifier_pipeline.classifier_trainer.config import TrainingConfig
from image_classifier_pipeline.lib.models import Dataset
from image_classifier_pipeline.lib.logger import setup_logger

logger = setup_logger(__name__)


class Trainer:
    """
    Trainer for a classifier.
    """

    def __init__(self, config: TrainingConfig, dataset: Dataset):
        self.config = config
        self.dataset = dataset

        logger.info(f"Dataset loaded with {len(dataset.train.items)} training images")

    def train(self):
        pass

    def evaluate(self):
        pass
