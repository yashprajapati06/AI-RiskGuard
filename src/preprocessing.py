"""Preprocessing used by the fraud model."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build the unfitted feature transformer."""
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    transformers = []
    if NUMERICAL_FEATURES:
        transformers.append(("numeric", numerical_pipeline, NUMERICAL_FEATURES))
    if CATEGORICAL_FEATURES:
        transformers.append(("categorical", categorical_pipeline, CATEGORICAL_FEATURES))
    if not transformers:
        raise ValueError("At least one model feature is required.")
    return ColumnTransformer(transformers=transformers, remainder="drop")
