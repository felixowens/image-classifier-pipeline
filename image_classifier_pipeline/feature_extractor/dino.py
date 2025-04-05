# Feature extractor plan

# 1. Load provided dataset file with (Dataset) class
# 2. Load the DINOmodel
# 3. Extract the features
# 4. Save the a new dataset file with an additional key "features"

from typing import Any
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

from image_classifier_pipeline.lib import setup_logger

logger = setup_logger(__name__)


class EvalPrediction:
    predictions: np.ndarray[Any, Any]
    label_ids: np.ndarray[Any, Any]


class DinoFeatureExtractor:
    """Extracts features from images using a DINO model."""

    model_name = "facebook/dinov2-base"

    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def _normalise_image(self, image: Image.Image):
        """
        Normalises the as expected by the DINO model.
        """
        pass

    def extract_features(self, image: Image.Image) -> torch.Tensor:
        """
        Takes an image and returns the features extracted by the DINO model.
        """
        inputs = self.processor(
            images=image, return_tensors="pt", padding=True, do_resize=True
        )
        outputs = self.model(**inputs)
        features: torch.Tensor = outputs.last_hidden_state
        single_feature: torch.Tensor = features.mean(dim=1)

        # Assert is a tensor
        assert isinstance(single_feature, torch.Tensor)

        return single_feature

    def save_features(self, output_dir: str):
        pass
