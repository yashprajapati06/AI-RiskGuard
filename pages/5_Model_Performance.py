"""Held-out model metrics, comparison, and global importance."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config import DATA_PATH, MODEL_METADATA_PATH
from src.monitoring import get_monitoring_summary
from src.ui_helpers import configure_page, ensure_app_ready, render_header
from src.utils import read_json, validate_model_metadata

configure_page("Model Performance", "🧠")
ensure_app_ready()
render_header(
    "🧠 Model Performance",
    "Reproducible held-out test metrics from the latest local training run.",
)

if not MODEL_METADATA_PATH.exists():
    st.error(
        "Training metadata is unavailable. Run `python -m src.train_model`, then "
        "reload this page."
    )
    st.stop()

try:
    metadata = read_json(MODEL_METADATA_PATH)
    validate_model_metadata(metadata)
except (OSError, ValueError, TypeError) as exc:
    st.error(
        "Training metadata exists but is incomplete or malformed. Retrain with "
        f"`python -m src.train_model`. Technical detail: {exc}"
    )
    st.stop()
selected_name = metadata["selected_model"]
selected_metrics = metadata["models"][selected_name]

st.subheader("Training summary")
top_columns = st.columns(4)
top_columns[0].metric("Model", selected_name.replace("_", " ").title())
top_columns[1].metric("Dataset Size", f"{metadata['dataset_size']:,}")
top_columns[2].metric("Fraud Rate", f"{metadata['fraud_rate'] * 100:.2f}%")
top_columns[3].metric("Training Date (UTC)", metadata["training_timestamp"][:10])
st.caption(
    f"Data source: {metadata['dataset_source']} · "
    f"Origin: {metadata['data_origin'].replace('_', ' ')} · "
    f"Split: {metadata['split_strategy'].replace('_', ' ')} · "
    f"ML features: {len(metadata['feature_list'])}"
)

with st.expander("Data provenance and leakage controls"):
    st.markdown(
        f"""
        - **Upstream source:** [{metadata["dataset_source"]}]({metadata["dataset_source_url"]})
        - **Source population:** {metadata["source_dataset_rows"]:,} fully synthetic events;
          the prepared model sample contains {metadata["dataset_size"]:,} rows.
        - **Sampling:** {metadata["source_sampling_strategy"].replace("_", " ")};
          the fraud target was not used to select rows.
        - **Amount handling:** {metadata["amount_normalization"]}
        - **Outer evaluation:** earliest {metadata["training_rows"]:,} rows for training,
          latest {metadata["test_rows"]:,} rows for locked chronological testing.
        - **Feature boundary:** IDs, timestamp, target, and manual-only inputs never enter
          either classifier. Historical aggregates are calculated from earlier events only.

        The source is fully synthetic—not real or anonymized customer data. TabFormer is
        the dataset source; this project trains Logistic Regression and Random Forest,
        not the TabFormer transformer architecture.
        """
    )

st.markdown("#### Fraud-focused evaluation metrics")
metric_columns = st.columns(4)
for column, (label, key) in zip(
    metric_columns,
    [
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("ROC-AUC", "roc_auc"),
    ],
):
    column.metric(label, f"{selected_metrics[key]:.3f}")

diagnostic_columns = st.columns(3)
for column, (label, key) in zip(
    diagnostic_columns,
    [
        ("Accuracy (context only)", "accuracy"),
        ("False Positive Rate", "false_positive_rate"),
        ("False Negative Rate", "false_negative_rate"),
    ],
):
    column.metric(label, f"{selected_metrics[key]:.3f}")

st.caption(f"Selection rule: {metadata['selection_reason']}")
st.warning(
    "Fraud is the minority class, so accuracy can look high even when a model misses "
    "fraud. Tuning therefore emphasizes recall and F1 while limiting mean "
    "cross-validation false positives to the documented review-capacity threshold."
)
st.info(
    f"Hyperparameters and the winning model were selected with "
    f"{metadata['cv_folds']}-fold cross-validation on training rows only. The final "
    f"metrics below come from the untouched {metadata['test_rows']:,}-row test "
    "partition."
)
st.info(
    "The saved class-weighted classifier's `predict_proba()` output is presented as "
    "an uncalibrated fraud-likelihood estimate. It supports relative ranking and the "
    "prototype risk score, but should not be interpreted as a real-world fraud rate."
)

with st.expander("What do these metrics mean?"):
    st.markdown(
        """
        - **Precision:** among transactions predicted as fraud, the fraction actually fraudulent.
        - **Recall:** among all fraudulent transactions, the fraction detected by the model.
        - **F1-score:** harmonic balance between precision and recall.
        - **ROC-AUC:** how well model likelihood scores rank fraud above genuine cases across thresholds.
        - **False Positive Rate:** genuine transactions incorrectly flagged.
        - **False Negative Rate:** fraud transactions incorrectly treated as genuine.

        In a payment-risk use case, missed fraud may cause loss, while excessive false
        positives may inconvenience genuine customers. The preferred trade-off depends
        on review capacity and business cost; this project uses an educational rule.
        """
    )

comparison = pd.DataFrame(metadata["models"]).T.reset_index(names="model")
comparison["model"] = comparison["model"].str.replace("_", " ").str.title()
comparison_columns = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "false_positive_rate",
    "false_negative_rate",
    "cv_selection_score",
    "selection_score",
]
st.subheader("Tuned Logistic Regression vs Random Forest")
st.dataframe(comparison[comparison_columns].round(4), hide_index=True, width="stretch")
st.caption(
    "CV selection score determines the saved model. Test selection score is shown "
    "only as a held-out diagnostic and does not influence model selection."
)

with st.expander("Selected hyperparameters"):
    for model_name, metrics in metadata["models"].items():
        st.markdown(f"**{model_name.replace('_', ' ').title()}**")
        st.json(metrics["best_parameters"])

left, right = st.columns([1, 1.25])
with left:
    matrix = selected_metrics["confusion_matrix"]
    figure = px.imshow(
        matrix,
        x=["Predicted Genuine", "Predicted Fraud"],
        y=["Actual Genuine", "Actual Fraud"],
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion matrix (held-out test set)",
        labels={"x": "Prediction", "y": "Actual", "color": "Count"},
    )
    st.plotly_chart(figure, width="stretch")
with right:
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    st.markdown(
        f"""
        #### Reading the matrix

        Rows show **actual labels** and columns show **model predictions**.

        - **TN = {tn:,}:** genuine transaction correctly identified.
        - **FP = {fp:,}:** genuine transaction incorrectly flagged.
        - **FN = {fn:,}:** fraud transaction incorrectly treated as genuine.
        - **TP = {tp:,}:** fraud transaction correctly detected.

        High false negatives may cause fraud losses. Excessive false positives
        can inconvenience genuine customers. The matrix is from the held-out test
        split—not from unlabeled live/manual transactions.
        """
    )

importance = pd.DataFrame(metadata.get("global_feature_importance", [])).head(15)
if not importance.empty:
    importance["feature"] = (
        importance["feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
    )
    figure = px.bar(
        importance.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Global feature importance (selected model)",
        labels={"importance": "Importance magnitude", "feature": "Transformed feature"},
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Global importance describes aggregate model behavior; it is not a per-transaction causal explanation."
    )

monitoring = get_monitoring_summary()
st.subheader("Prediction monitoring")
monitor_columns = st.columns(3)
monitor_columns[0].metric(
    "Live Average Risk", f"{monitoring['average_prediction_risk']:.2f}"
)
monitor_columns[1].metric(
    "Live High-Risk Predictions", monitoring["high_risk_predictions"]
)
monitor_columns[2].metric("Model Version", monitoring["model_version"])
st.info(monitoring["live_accuracy_notice"])
st.caption(
    "Live monitoring summarizes prediction behavior only. It must not be presented "
    "as confirmed fraud detection performance without outcome labels."
)

with st.expander("Training dataset visual checks"):
    available_columns = pd.read_csv(DATA_PATH, nrows=0).columns
    visual_columns = ["fraud", "amount", "hour_of_day"]
    if "payment_method" in available_columns:
        visual_columns.append("payment_method")
    data = pd.read_csv(DATA_PATH, usecols=visual_columns)
    col1, col2 = st.columns(2)
    with col1:
        label_counts = (
            data["fraud"].map({0: "Genuine", 1: "Fraud"}).value_counts().reset_index()
        )
        label_counts.columns = ["Class", "Count"]
        st.plotly_chart(
            px.bar(
                label_counts,
                x="Class",
                y="Count",
                color="Class",
                title="Fraud vs genuine distribution",
            ),
            width="stretch",
        )
    with col2:
        if "payment_method" in data and data["payment_method"].nunique() > 1:
            group_column = "payment_method"
            chart_title = "Fraud rate by payment method"
            axis_label = "Payment method"
        else:
            group_column = "hour_of_day"
            chart_title = "Fraud rate by transaction hour"
            axis_label = "Hour of day"
        rates = data.groupby(group_column, as_index=False)["fraud"].mean()
        rates["fraud_rate_percent"] = rates["fraud"] * 100
        st.plotly_chart(
            px.bar(
                rates,
                x=group_column,
                y="fraud_rate_percent",
                title=chart_title,
                labels={
                    group_column: axis_label,
                    "fraud_rate_percent": "Fraud rate (%)",
                },
            ),
            width="stretch",
        )
    histogram_data = (
        data.sample(n=100_000, random_state=42) if len(data) > 100_000 else data
    )
    st.plotly_chart(
        px.histogram(
            histogram_data,
            x="amount",
            color=histogram_data["fraud"].map({0: "Genuine", 1: "Fraud"}),
            nbins=60,
            title="Transaction amount distribution",
            labels={"color": "Class", "amount": "Amount (₹)"},
        ),
        width="stretch",
    )
