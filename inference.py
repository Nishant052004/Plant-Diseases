"""Small, testable helpers for the saved plant-disease classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

IMAGE_SIZE = (224, 224)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert an uploaded image to the model's expected RGB float batch."""
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image.")

    rgb_image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    pixels = np.asarray(rgb_image, dtype=np.float32) / 255.0
    return np.expand_dims(pixels, axis=0)


def prediction_probabilities(raw_prediction: Any, class_count: int) -> np.ndarray:
    """Validate a model output and return one finite probability per class."""
    values = np.asarray(raw_prediction, dtype=np.float32)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 1 or values.shape[0] != class_count:
        raise ValueError(
            f"Model returned {values.shape} for {class_count} labels; "
            "the model and labels.txt do not match."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Model returned non-finite prediction values.")

    # Saved classifiers normally return softmax probabilities. Normalize
    # logits as a fallback so inference remains correct for an uncompiled
    # classifier or a model exported without its final activation.
    if np.any(values < 0) or np.any(values > 1) or not np.isclose(
        float(values.sum()), 1.0, atol=1e-3
    ):
        values = values - np.max(values)
        values = np.exp(values)
        values /= values.sum()
    return values


def top_predictions(probabilities: np.ndarray, limit: int = 3) -> list[tuple[int, float]]:
    """Return up to ``limit`` class indexes and probabilities in descending order."""
    if limit < 1:
        raise ValueError("Prediction limit must be positive.")
    indexes = np.argsort(probabilities)[::-1][:limit]
    return [(int(index), float(probabilities[index])) for index in indexes]


def project_path(filename: str) -> Path:
    """Resolve an application asset independently of the current directory."""
    return Path(__file__).resolve().parent / filename
