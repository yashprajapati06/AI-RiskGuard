"""Safe, idempotent project initialization for Streamlit and scripts."""

from __future__ import annotations

import logging
import pickle
import zlib
from typing import Any

import joblib

from config import (
    ARTIFACT_SCHEMA_VERSION,
    DATA_PATH,
    MODEL_FEATURES,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    NON_MODEL_INPUT_FEATURES,
    PREPROCESSOR_PATH,
    ensure_directories,
)
from src.data_generator import generate_synthetic_transactions
from src.database import initialize_database
from src.evaluation import get_fraud_class_index
from src.utils import read_json, validate_model_metadata

LOGGER = logging.getLogger(__name__)


def model_artifacts_are_valid() -> bool:
    """Check that the model, preprocessor, and required metadata are loadable."""
    if not all(
        path.exists() for path in (MODEL_PATH, PREPROCESSOR_PATH, MODEL_METADATA_PATH)
    ):
        return False
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        metadata = read_json(MODEL_METADATA_PATH)
        validate_model_metadata(metadata)
        get_fraud_class_index(model)
        transformed_feature_count = len(preprocessor.get_feature_names_out())
        return (
            callable(getattr(model, "predict_proba", None))
            and callable(getattr(preprocessor, "transform", None))
            and hasattr(preprocessor, "transformers_")
            and list(getattr(preprocessor, "feature_names_in_", [])) == MODEL_FEATURES
            and metadata["model_version"] == ARTIFACT_SCHEMA_VERSION
            and metadata["feature_list"] == MODEL_FEATURES
            and metadata["non_model_input_features"] == NON_MODEL_INPUT_FEATURES
            and int(getattr(model, "n_features_in_", -1)) == transformed_feature_count
            and int(metadata.get("transformed_feature_count", -1))
            == transformed_feature_count
        )
    except (
        OSError,
        EOFError,
        ValueError,
        TypeError,
        AttributeError,
        ImportError,
        IndexError,
        KeyError,
        pickle.PickleError,
        zlib.error,
    ):
        LOGGER.warning("Model artifact validation failed; retraining is required.")
        return False


def initialize_project() -> dict[str, Any]:
    """Ensure required local data, model artifacts, and database exist."""
    ensure_directories()
    generated_data = False
    trained_model = False
    if not DATA_PATH.exists():
        generate_synthetic_transactions()
        generated_data = True
    if not model_artifacts_are_valid():
        from src.train_model import train_and_save_models

        train_and_save_models()
        trained_model = True
    initialize_database()
    return {
        "generated_data": generated_data,
        "trained_model": trained_model,
    }
