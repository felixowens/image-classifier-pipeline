# Feature extractor plan

# 1. Load provided dataset file with (Dataset) class
# 2. Load the DINOmodel
# 3. Extract the features
# 4. Save the a new dataset file with an additional key "features"


from image_classifier_pipeline.lib import setup_logger
from image_classifier_pipeline.lib.models import Dataset

logger = setup_logger(__name__)


class DinoFeatureExtractor:
    def __init__(self, dataset_path: str):
        self.dataset = Dataset.load_from_saved_folder(dataset_path)

        logger.info(f"Loaded dataset: {self.dataset}")

    def extract_features(self):
        pass

    def save_features(self, output_dir: str):
        pass
