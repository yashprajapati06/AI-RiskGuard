"""Fraud model training and selection."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from config import (
    ARTIFACT_SCHEMA_VERSION,
    BASE_DIR,
    CV_TUNING_MAX_ROWS,
    DATA_PATH,
    DATASET_METADATA_PATH,
    EVENT_TIMESTAMP_COLUMN,
    MODEL_FEATURES,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    MODEL_SELECTION_WEIGHTS,
    MODELS_DIR,
    NON_MODEL_INPUT_FEATURES,
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
from src.utils import (
    atomic_joblib_dump,
    configure_logging,
    file_sha256,
    read_json,
    write_json,
)
from src.validation import validate_training_dataset

LOGGER = logging.getLogger(__name__)

CV_FOLDS = 5
MAX_CV_FALSE_POSITIVE_RATE = 0.05

CV_SCORING = {
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "specificity": make_scorer(recall_score, pos_label=0, zero_division=0),
}


def _project_relative(path: Any) -> str:
    """Format a path for console output."""
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
            # Keep the demo artifact small enough for GitHub and Streamlit Cloud.
            n_estimators=120,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def _parameter_grids(target: pd.Series) -> dict[str, dict[str, list[Any]]]:
    """Build class-weight options from the training imbalance."""
    counts = target.value_counts()
    if not {0, 1}.issubset(counts.index):
        raise ValueError("Class-weight tuning requires both target classes.")
    imbalance_ratio = float(counts[0] / counts[1])
    numeric_ratios = {9, 16}
    numeric_ratios.update(
        max(2, round(imbalance_ratio * multiplier)) for multiplier in (0.125, 0.25, 0.5)
    )
    flattened_ratios = sorted(numeric_ratios)
    logistic_weights: list[Any] = ["balanced"] + [
        {0: 1, 1: ratio} for ratio in flattened_ratios
    ]
    forest_weights: list[Any] = ["balanced_subsample"] + [
        {0: 1, 1: ratio} for ratio in flattened_ratios
    ]
    return {
        "logistic_regression": {
            "classifier__C": [0.03, 0.1, 0.3, 1.0],
            "classifier__class_weight": logistic_weights,
            "classifier__solver": ["liblinear"],
        },
        "random_forest": {
            "classifier__class_weight": forest_weights,
            "classifier__max_depth": [8, 12],
            "classifier__min_samples_leaf": [5, 20],
        },
    }


def _cv_composite_scores(cv_results: dict[str, Any]) -> np.ndarray:
    """Calculate the selection score for each CV candidate."""
    scores = np.zeros(len(cv_results["params"]), dtype=float)
    for metric_name, weight in MODEL_SELECTION_WEIGHTS.items():
        scores += np.asarray(cv_results[f"mean_test_{metric_name}"]) * weight
    return scores


def _select_cv_candidate(cv_results: dict[str, Any]) -> int:
    """Pick the best CV result that stays under the false-positive limit."""
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
    """Collect mean CV metrics for the selected parameters."""
    index = int(search.best_index_)
    summary = {
        metric_name: float(search.cv_results_[f"mean_test_{metric_name}"][index])
        for metric_name in ("precision", "recall", "f1", "roc_auc", "specificity")
    }
    summary["false_positive_rate"] = 1.0 - summary["specificity"]
    summary["selection_score"] = model_selection_score(summary)
    return summary


def _clean_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Strip Pipeline prefixes before saving parameters."""
    return {
        key.removeprefix("classifier__"): value for key, value in parameters.items()
    }


def split_training_dataset(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    dict[str, Any],
]:
    """Engineer features and make the outer 80/20 split.

    Timestamped data is split chronologically. The generated fallback has no
    timestamp, so it uses a reproducible stratified split.
    """
    engineered = engineer_features(dataframe)
    features = engineered[MODEL_FEATURES]
    target = engineered["fraud"].astype(int)

    if EVENT_TIMESTAMP_COLUMN in engineered.columns:
        timestamps = pd.to_datetime(
            engineered[EVENT_TIMESTAMP_COLUMN], errors="coerce", utc=True
        )
        if timestamps.isna().any():
            raise ValueError(f"{EVENT_TIMESTAMP_COLUMN} must contain valid timestamps.")
        ordered_positions = np.argsort(timestamps.to_numpy(), kind="stable")
        split_index = int(len(engineered) * (1.0 - TEST_SIZE))
        if split_index <= 0 or split_index >= len(engineered):
            raise ValueError("The chronological split produced an empty partition.")
        training_positions = ordered_positions[:split_index]
        test_positions = ordered_positions[split_index:]
        x_train = features.iloc[training_positions]
        x_test = features.iloc[test_positions]
        y_train = target.iloc[training_positions]
        y_test = target.iloc[test_positions]
        train_times = timestamps.iloc[training_positions]
        test_times = timestamps.iloc[test_positions]
        split_metadata = {
            "split_strategy": "chronological_80_20",
            "training_period_start": train_times.min().isoformat(),
            "training_period_end": train_times.max().isoformat(),
            "test_period_start": test_times.min().isoformat(),
            "test_period_end": test_times.max().isoformat(),
        }
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=target,
        )
        split_metadata = {"split_strategy": "stratified_random_80_20"}

    for partition_name, partition_target in (
        ("training", y_train),
        ("test", y_test),
    ):
        if set(partition_target.unique().tolist()) != {0, 1}:
            raise ValueError(
                f"The {partition_name} partition must contain both target classes."
            )
    return engineered, x_train, x_test, y_train, y_test, split_metadata


def _training_tuning_sample(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Take the stratified tuning sample from training rows only."""
    if len(features) <= CV_TUNING_MAX_ROWS:
        return features, target
    tuning_features, _, tuning_target, _ = train_test_split(
        features,
        target,
        train_size=CV_TUNING_MAX_ROWS,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    return tuning_features, tuning_target


def _load_dataset_provenance(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Load source details from the dataset manifest."""
    if not DATASET_METADATA_PATH.exists():
        return {
            "source_id": "unverified_local_dataset",
            "source_name": "Local training dataset (source manifest unavailable)",
            "source_url": "local:data.transactions",
            "data_origin": "unverified_local_dataset",
            "source_rows": len(dataframe),
            "sample_rows": len(dataframe),
            "sampling_strategy": "unknown",
            "amount_normalization": "Unknown; source manifest unavailable.",
        }
    provenance = read_json(DATASET_METADATA_PATH)
    required = {
        "source_id",
        "source_name",
        "source_url",
        "data_origin",
        "source_rows",
        "sample_rows",
        "sampling_strategy",
        "amount_normalization",
    }
    missing = required.difference(provenance)
    if missing:
        raise ValueError("Dataset metadata is missing: " + ", ".join(sorted(missing)))
    if int(provenance["sample_rows"]) != len(dataframe):
        raise ValueError(
            "Dataset metadata sample_rows does not match the training CSV."
        )
    if int(provenance["source_rows"]) < len(dataframe):
        raise ValueError(
            "Dataset metadata source_rows cannot be smaller than sample_rows."
        )
    expected_digest = provenance.get("dataset_sha256")
    if expected_digest is not None:
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise ValueError("Dataset metadata contains an invalid SHA-256 digest.")
        if file_sha256(DATA_PATH) != expected_digest.casefold():
            raise ValueError(
                "Dataset hash does not match data/dataset_metadata.json. "
                "Regenerate the processed dataset before training."
            )
    return provenance


def train_and_save_models() -> dict[str, Any]:
    """Tune, compare, and save the candidate models.

    Model choice happens inside training-only CV. The held-out split is evaluated
    after refitting and never affects selection.
    """
    configure_logging()
    ensure_directories()
    dataframe = load_or_generate_dataset(DATA_PATH)
    validate_training_dataset(dataframe)
    provenance = _load_dataset_provenance(dataframe)
    (
        engineered,
        x_train,
        x_test,
        y_train,
        y_test,
        split_metadata,
    ) = split_training_dataset(dataframe)
    target = engineered["fraud"].astype(int)
    tuning_features, tuning_target = _training_tuning_sample(x_train, y_train)

    # Only MODEL_FEATURES reaches the classifiers; IDs, timestamps, rule-only
    # fields, and the fraud target stay out.
    tuning_counts = tuning_target.value_counts()
    if len(tuning_counts) < 2 or tuning_counts.min() < CV_FOLDS:
        raise ValueError(
            "The training-only tuning sample needs at least five rows from each "
            "target class."
        )

    models = _build_models()
    parameter_grids = _parameter_grids(tuning_target)
    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    evaluations: dict[str, dict[str, Any]] = {}
    fitted_pipelines: dict[str, Pipeline] = {}
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
        search.fit(tuning_features, tuning_target)
        # Tune on a manageable training-only sample, then refit on all training rows.
        fitted_pipeline = clone(pipeline).set_params(**search.best_params_)
        fitted_pipeline.fit(x_train, y_train)
        fitted_pipelines[model_name] = fitted_pipeline

        # Score the refitted pipeline once on the held-out set.
        evaluations[model_name] = evaluate_classifier(fitted_pipeline, x_test, y_test)
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

    # Test metrics are reporting only; CV results decide which model is saved.
    selected_name = max(
        fitted_pipelines,
        key=lambda name: (
            evaluations[name]["cv_selection_score"],
            evaluations[name]["cv_metrics"]["roc_auc"],
            evaluations[name]["cv_metrics"]["precision"],
        ),
    )
    selected_metrics = evaluations[selected_name]
    selected_pipeline = fitted_pipelines[selected_name]
    preprocessor = selected_pipeline.named_steps["preprocessor"]
    selected_model = selected_pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    feature_importance = extract_feature_importance(selected_model, feature_names)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Compression keeps the checked-in artifacts reasonably small.
    atomic_joblib_dump(selected_model, MODEL_PATH, compress=3)
    atomic_joblib_dump(preprocessor, PREPROCESSOR_PATH, compress=3)
    dataset_digest = file_sha256(DATA_PATH)
    model_digest = file_sha256(MODEL_PATH)
    preprocessor_digest = file_sha256(PREPROCESSOR_PATH)

    timestamp = datetime.now(timezone.utc).isoformat()
    metadata: dict[str, Any] = {
        "model_version": ARTIFACT_SCHEMA_VERSION,
        "selected_model": selected_name,
        "selection_reason": (
            "Selected using five-fold cross-validation on a bounded sample drawn "
            "only from the training partition, then refitted on all training rows. "
            "Candidates must keep mean validation false-positive rate at or below "
            f"{MAX_CV_FALSE_POSITIVE_RATE:.0%}, then "
            "maximize 35% recall, 35% F1, 20% ROC-AUC, and 10% precision. "
            "The held-out test set is used only for final reporting."
        ),
        "training_timestamp": timestamp,
        "dataset_size": len(engineered),
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "tuning_rows": len(tuning_features),
        "fraud_rate": float(target.mean()),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "cv_strategy": "stratified_shuffled_training_only_sample",
        "maximum_cv_false_positive_rate": MAX_CV_FALSE_POSITIVE_RATE,
        "selection_partition": "training_only_cross_validation",
        "refit_rows": len(x_train),
        **split_metadata,
        "dataset_source_id": str(provenance["source_id"]),
        "dataset_source": str(provenance["source_name"]),
        "dataset_source_url": str(provenance["source_url"]),
        "data_origin": str(provenance["data_origin"]),
        "source_dataset_rows": int(provenance["source_rows"]),
        "source_sampling_strategy": str(provenance["sampling_strategy"]),
        "amount_normalization": str(provenance["amount_normalization"]),
        "dataset_sha256": dataset_digest,
        "model_sha256": model_digest,
        "preprocessor_sha256": preprocessor_digest,
        "upstream_license": str(provenance.get("upstream_license", "not_recorded")),
        "eligible_positive_purchase_rows": int(
            provenance.get("eligible_positive_purchase_rows", len(engineered))
        ),
        "excluded_non_positive_rows": int(
            provenance.get("excluded_non_positive_rows", 0)
        ),
        "amount_filter": str(provenance.get("amount_filter", "not_recorded")),
        "feature_list": MODEL_FEATURES,
        "non_model_input_features": NON_MODEL_INPUT_FEATURES,
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
