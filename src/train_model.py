"""Train, compare, select, and persist fraud-detection models."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from config import (
    BASE_DIR,
    DATA_PATH,
    MODEL_FEATURES,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    MODEL_SELECTION_WEIGHTS,
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
)
from src.feature_engineering import engineer_features
from src.preprocessing import build_preprocessor
from src.utils import configure_logging, write_json
from src.validation import validate_training_dataset

LOGGER = logging.getLogger(__name__)

CV_FOLDS = 5
MAX_CV_FALSE_POSITIVE_RATE = 0.20

CV_SCORING = {
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "specificity": make_scorer(recall_score, pos_label=0, zero_division=0),
}


def _project_relative(path: Any) -> str:
    """Return a console-safe path relative to the project when possible."""
    try:
        return path.resolve().relative_to(BASE_DIR.resolve()).as_posix()
    except ValueError:
        return path.name


def _build_models() -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2_000,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=320,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def _parameter_grids() -> dict[str, dict[str, list[Any]]]:
    """Return bounded grids selected for the existing two candidate models."""
    logistic_weights = [{0: 1, 1: ratio} for ratio in range(9, 17)]
    forest_weights: list[Any] = [
        "balanced_subsample",
        {0: 1, 1: 15},
        {0: 1, 1: 25},
    ]
    return {
        "logistic_regression": {
            "classifier__C": [0.03, 0.1, 0.3, 1.0, 3.0],
            "classifier__class_weight": logistic_weights,
            "classifier__solver": ["liblinear", "lbfgs"],
        },
        "random_forest": {
            "classifier__class_weight": forest_weights,
            "classifier__max_depth": [8, 14, None],
            "classifier__min_samples_leaf": [2, 5, 10],
        },
    }


def _cv_composite_scores(cv_results: dict[str, Any]) -> np.ndarray:
    """Calculate the documented composite for every CV configuration."""
    scores = np.zeros(len(cv_results["params"]), dtype=float)
    for metric_name, weight in MODEL_SELECTION_WEIGHTS.items():
        scores += np.asarray(cv_results[f"mean_test_{metric_name}"]) * weight
    return scores


def _select_cv_candidate(cv_results: dict[str, Any]) -> int:
    """Choose the strongest CV candidate under the false-positive constraint."""
    composite_scores = _cv_composite_scores(cv_results)
    specificity = np.asarray(cv_results["mean_test_specificity"], dtype=float)
    false_positive_rates = 1.0 - specificity
    eligible = np.flatnonzero(
        false_positive_rates <= MAX_CV_FALSE_POSITIVE_RATE + 1e-12
    )
    if not len(eligible):
        raise ValueError(
            "No cross-validation configuration satisfies the false-positive-rate "
            f"limit of {MAX_CV_FALSE_POSITIVE_RATE:.0%}."
        )
    return int(
        max(
            eligible,
            key=lambda index: (
                composite_scores[index],
                cv_results["mean_test_roc_auc"][index],
                cv_results["mean_test_precision"][index],
            ),
        )
    )


def _cv_summary(search: GridSearchCV) -> dict[str, float]:
    """Return serializable mean validation metrics for the chosen configuration."""
    index = int(search.best_index_)
    summary = {
        metric_name: float(search.cv_results_[f"mean_test_{metric_name}"][index])
        for metric_name in ("precision", "recall", "f1", "roc_auc", "specificity")
    }
    summary["false_positive_rate"] = 1.0 - summary["specificity"]
    summary["selection_score"] = model_selection_score(summary)
    return summary


def _clean_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Remove Pipeline prefixes before parameters are written to metadata."""
    return {
        key.removeprefix("classifier__"): value for key, value in parameters.items()
    }


def train_and_save_models() -> dict[str, Any]:
    """Tune, train, compare, and persist candidates without test-set leakage.

    Hyperparameters and model selection use stratified cross-validation inside the
    training partition. The locked test partition is transformed only by the final
    refitted pipelines and is used exclusively for final performance reporting.
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

    models = _build_models()
    parameter_grids = _parameter_grids()
    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    evaluations: dict[str, dict[str, Any]] = {}
    fitted_searches: dict[str, GridSearchCV] = {}
    for model_name, model in models.items():
        LOGGER.info("Tuning %s with %d-fold cross-validation", model_name, CV_FOLDS)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", model),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grids[model_name],
            scoring=CV_SCORING,
            refit=_select_cv_candidate,
            cv=cross_validation,
            n_jobs=-1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(x_train, y_train)
        fitted_searches[model_name] = search

        # Held-out metrics are calculated only after CV tuning has finished.
        evaluations[model_name] = evaluate_classifier(
            search.best_estimator_, x_test, y_test
        )
        evaluations[model_name]["selection_score"] = model_selection_score(
            evaluations[model_name]
        )
        validation_metrics = _cv_summary(search)
        evaluations[model_name]["cv_metrics"] = validation_metrics
        evaluations[model_name]["cv_selection_score"] = validation_metrics[
            "selection_score"
        ]
        evaluations[model_name]["best_parameters"] = _clean_parameters(
            search.best_params_
        )

    # Selection uses training-only CV results; the test metrics above cannot affect
    # which model is persisted.
    selected_name = max(
        fitted_searches,
        key=lambda name: (
            evaluations[name]["cv_selection_score"],
            evaluations[name]["cv_metrics"]["roc_auc"],
            evaluations[name]["cv_metrics"]["precision"],
        ),
    )
    selected_metrics = evaluations[selected_name]
    selected_pipeline = fitted_searches[selected_name].best_estimator_
    preprocessor = selected_pipeline.named_steps["preprocessor"]
    selected_model = selected_pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    feature_importance = extract_feature_importance(selected_model, feature_names)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    timestamp = datetime.now(timezone.utc).isoformat()
    metadata: dict[str, Any] = {
        "model_version": "1.1.0",
        "selected_model": selected_name,
        "selection_reason": (
            "Selected using five-fold training-only cross-validation. Candidates "
            "must keep mean validation false-positive rate at or below 20%, then "
            "maximize 35% recall, 35% F1, 20% ROC-AUC, and 10% precision. "
            "The held-out test set is used only for final reporting."
        ),
        "training_timestamp": timestamp,
        "dataset_size": len(engineered),
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "fraud_rate": float(target.mean()),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "maximum_cv_false_positive_rate": MAX_CV_FALSE_POSITIVE_RATE,
        "selection_partition": "training_only_cross_validation",
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
            "cv_selection_score",
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
