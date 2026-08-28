"""Train, compare, select, and persist fraud-detection models."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from config import (
    BASE_DIR,
    DATA_PATH,
    MODEL_FEATURES,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    ensure_directories,
)
from src.data_generator import load_or_generate_dataset
from src.evaluation import (
    evaluate_classifier,
    extract_feature_importance,
    model_selection_score,
    select_best_model,
)
from src.feature_engineering import engineer_features
from src.preprocessing import build_preprocessor
from src.utils import configure_logging, write_json
from src.validation import validate_training_dataset

LOGGER = logging.getLogger(__name__)


def _project_relative(path: Any) -> str:
    """Return a console-safe path relative to the project when possible."""
    try:
        return path.resolve().relative_to(BASE_DIR.resolve()).as_posix()
    except ValueError:
        return path.name


def _build_models() -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1_500,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=240,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train_and_save_models() -> dict[str, Any]:
    """Train both candidates without leakage and persist the selected artifacts.

    Preprocessing is fitted on the training partition only. The returned metadata
    contains measured held-out metrics for both candidates and the documented
    model-selection result.
    """
    configure_logging()
    ensure_directories()
    dataframe = load_or_generate_dataset(DATA_PATH)
    validate_training_dataset(dataframe)
    engineered = engineer_features(dataframe)

    # MODEL_FEATURES is an explicit allow-list: IDs and the fraud target are never
    # passed to either candidate model.
    features = engineered[MODEL_FEATURES]
    target = engineered["fraud"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    preprocessor = build_preprocessor()
    # The transformer sees only training rows; test rows remain fully held out.
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)

    models = _build_models()
    evaluations: dict[str, dict[str, Any]] = {}
    for model_name, model in models.items():
        LOGGER.info("Training %s", model_name)
        model.fit(x_train_transformed, y_train)
        evaluations[model_name] = evaluate_classifier(model, x_test_transformed, y_test)
        evaluations[model_name]["selection_score"] = model_selection_score(
            evaluations[model_name]
        )

    selected_name, selected_metrics = select_best_model(evaluations)
    selected_model = models[selected_name]
    feature_names = preprocessor.get_feature_names_out()
    feature_importance = extract_feature_importance(selected_model, feature_names)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    timestamp = datetime.now(timezone.utc).isoformat()
    metadata: dict[str, Any] = {
        "model_version": "1.0.0",
        "selected_model": selected_name,
        "selection_reason": (
            "Highest composite score: 35% recall, 35% F1, 20% ROC-AUC, "
            "and 10% precision. This prioritizes finding fraud while monitoring "
            "false positives."
        ),
        "training_timestamp": timestamp,
        "dataset_size": len(engineered),
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "fraud_rate": float(target.mean()),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "feature_list": MODEL_FEATURES,
        "transformed_feature_count": len(feature_names),
        "models": evaluations,
        "selected_model_metrics": selected_metrics,
        "global_feature_importance": feature_importance,
    }
    write_json(MODEL_METADATA_PATH, metadata)

    print(f"Dataset loaded: {len(engineered):,} rows")
    print(f"Fraud rate: {target.mean() * 100:.2f}%")
    print("\nModel comparison:")
    comparison = pd.DataFrame(evaluations).T[
        [
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "false_positive_rate",
            "selection_score",
        ]
    ]
    print(comparison.round(4).to_string())
    print(f"\nSelected model: {selected_name}")
    print(f"Saved model: {_project_relative(MODEL_PATH)}")
    print(f"Saved preprocessor: {_project_relative(PREPROCESSOR_PATH)}")
    print(f"Saved metadata: {_project_relative(MODEL_METADATA_PATH)}")
    LOGGER.info("Selected and saved %s", selected_name)
    return metadata


def main() -> None:
    train_and_save_models()


if __name__ == "__main__":
    main()
