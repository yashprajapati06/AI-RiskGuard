# AI RiskGuard

AI RiskGuard is a local Streamlit workspace for exploring payment-risk scoring.
It combines a class-weighted machine-learning model with a small, readable rule
layer, stores assessments in SQLite, and surfaces high-risk records for manual
review.

**[Open the Streamlit demo](https://ai-riskguard-payment.streamlit.app/)**

> This is an educational prototype built with fully synthetic data. It is not an
> official Razorpay, IBM, or TabFormer product, and it must not be used to approve,
> decline, or authorize real payments.

## What is included

- A transaction checker with Normal, Medium, and High sample profiles
- Shared validation and feature engineering for training and prediction
- Logistic Regression and Random Forest training with imbalance-aware tuning
- A combined model-and-rules risk score with visible rule triggers
- A monitoring dashboard, searchable transaction log, and high-risk alert queue
- Reproducible data provenance, model metadata, artifact checksums, and tests

The interface accepts behavioral fields only. Card numbers, CVVs, OTPs, PINs,
bank passwords, and other payment credentials do not belong in this app.

## Quick start

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux or macOS
source .venv/bin/activate
```

Install the dependencies and start Streamlit:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The repository includes the prepared dataset and compatible model artifacts, so
a normal launch does not retrain the model. Startup verifies the dataset, model,
preprocessor, and metadata before loading them. Missing or mismatched artifacts
are rebuilt locally.

## Using the app

| Page | Purpose |
|---|---|
| Dashboard | Review saved assessment volume, risk mix, trends, and recent alerts |
| Transaction Risk Checker | Change a sample transaction, calculate its score, and save the result |
| Transaction Monitor | Filter saved records, inspect details, and export the current view |
| Risk Alerts | Review transactions that reached the HIGH threshold |
| Model Performance | Inspect held-out metrics, model selection, and global feature importance |

A practical walkthrough is to score each sample profile in the checker, then open
the dashboard and alert pages to see how the saved results flow through the app.

## System flow

```text
Transaction input
      |
      v
Validation and feature engineering
      |
      +-------------------+
      |                   |
      v                   v
Saved ML model       Transparent rules
      |                   |
      +---------+---------+
                v
        70% model + 30% rules
                |
        LOW / MEDIUM / HIGH
                |
      SQLite record + HIGH alert
                |
       Dashboard and monitor
```

The checker creates anonymous transaction, user, and merchant IDs for local
record lookup. Identifiers, timestamps, and the fraud target never enter the
model feature matrix.

## Risk score

The selected classifier produces an uncalibrated fraud-likelihood estimate. The
app converts it to a 0–100 model score and blends it with the normalized rule
score:

```text
model risk score = predict_proba output × 100
final risk score = 0.70 × model score + 0.30 × rule score
```

| Final score | Risk band | Review guidance |
|---:|---|---|
| Below 35 | LOW | Routine monitoring |
| 35 to below 70 | MEDIUM | Additional verification or manual review |
| 70 to 100 | HIGH | Immediate manual review |

The rules cover signals such as unusual amount, recent errors, transaction
velocity, new-device context, location change, merchant risk, international use,
and unusual hours. Their thresholds live in `config.py`. Rule messages explain
only the rule component; they are not causal explanations of the ML model.

## Dataset

The training data comes from the fully synthetic credit-card records published
with [IBM TabFormer](https://github.com/IBM/TabFormer) and mirrored as the
[Credit Card Transactions dataset on Kaggle](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions).
These are simulated records, not anonymized customer transactions.

The preparation pipeline in `src/tabformer_adapter.py`:

- reads 24,386,900 source transactions;
- removes 1,264,896 refunds, credits, and zero-value rows without consulting the
  fraud label;
- derives historical features from the remaining 23,122,004 positive payments;
- keeps a deterministic, label-independent 500,000-row sample using seed 42; and
- writes provenance and a SHA-256 digest to `data/dataset_metadata.json`.

The prepared sample contains 635 fraud labels, a natural fraud rate of 0.127%.
Historical averages, recent error counts, velocity, and location changes are
calculated before sampling so a retained row can use earlier eligible events.

Source amounts are multiplied by a fixed factor of `83.0` for the INR-oriented
demo. This is a reproducibility choice, not a live exchange rate.

### Feature boundaries

The ML model uses amount, recent error and velocity history, prior average amount,
location change, account age, hour, weekend status, international status, and five
derived risk features.

Some checker fields are intentionally outside the model:

- `is_new_device` and `merchant_risk_score` are manual, rule-only inputs because
  the source has no reliable equivalent.
- `payment_method` and `device_type` are stored as context but do not change the
  current model or rule score.
- `transaction_id`, `user_id`, `merchant_id`, `event_timestamp`, and `fraud` are
  excluded from model inputs.

The source `Errors?` field is only a proxy for failed activity, and location change
compares merchant locations rather than device GPS. More detail is available in
[`data/README.md`](data/README.md).

## Training and evaluation

Rows are ordered by `event_timestamp`. The earliest 400,000 rows form the training
partition; the latest 100,000 rows form the locked chronological test partition.
Hyperparameter search uses five-fold cross-validation on a deterministic 60,000-row
sample drawn only from the training period. Each chosen candidate is then refitted
on all 400,000 training rows.

The two candidates are:

- Logistic Regression with tuned regularization and fraud-class weight
- Random Forest with tuned depth, leaf size, and class weight

Candidate configurations must keep mean cross-validation false-positive rate at
or below 5%. Eligible configurations are ranked by 35% recall, 35% F1, 20%
ROC-AUC, and 10% precision. The locked test set does not influence selection.

### Current held-out results

Training-only cross-validation selected **Logistic Regression**. The following
metrics come from the 100,000-row chronological test partition:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 97.53% | 4.16% | 95.54% | 7.96% | 0.9733 | 2.47% | 4.46% |
| Random Forest | 95.87% | 0.86% | 31.25% | 1.67% | 0.8392 | 4.06% | 68.75% |

Selected-model confusion matrix (`[[TN, FP], [FN, TP]]`):

```text
[[97420, 2468],
 [    5,  107]]
```

The selected model found 107 of 112 fraud-labelled test rows, but it also flagged
2,468 genuine rows. Its **4.16% precision is a major limitation**. High accuracy
and recall should not be read as production readiness.

Class weighting also means the `predict_proba()` output is not calibrated as a
real-world fraud probability. The app uses it for relative ranking only. Exact
full-precision metrics and parameters are stored in
[`models/model_metadata.json`](models/model_metadata.json).

## Rebuild the dataset and model

The included artifacts are ready to use. To reproduce them, download the Kaggle
ZIP containing both the transactions and cards CSV files, then run from the
project root:

```powershell
python -m src.tabformer_adapter "C:\path\to\credit-card-transactions.zip" --target-rows 500000 --chunksize 200000 --seed 42
python -m src.train_model
```

The source ZIP is roughly 276 MB and its extracted transaction CSV is roughly
2.35 GB. Conversion and training can take substantial CPU, memory, and time. Do
not commit the archive or extracted source files. The small
`python -m src.data_generator` path is an offline fallback for tests or recovery;
models trained from it must not be described as TabFormer-derived.

Model and preprocessor writes are staged before replacement. Their SHA-256
digests, together with a line-ending-independent training-dataset digest, are
saved in metadata and checked at startup to catch stale, corrupted, or
mismatched artifacts across Windows and Linux deployments.

## Tests

Run the full suite:

```bash
python -m pytest
```

The suite covers validation, feature engineering, leakage boundaries, scoring,
model selection, artifact integrity, SQLite behavior, and the Streamlit checker.
Useful focused checks are:

```bash
python -m pytest tests/test_transaction_checker_qa.py -q
python -m compileall -q app.py config.py src pages tests
python -c "from src.bootstrap import initialize_project; print(initialize_project())"
```

## Project map

| Path | Responsibility |
|---|---|
| `app.py`, `pages/` | Streamlit interface |
| `src/tabformer_adapter.py` | Source conversion, historical features, and sampling |
| `src/train_model.py` | Split, tune, refit, evaluate, and persist models |
| `src/predictor.py` | Reusable transaction-scoring service |
| `src/rule_engine.py`, `src/risk_engine.py` | Rule checks and final risk band |
| `src/database.py`, `src/monitoring.py` | SQLite records, alerts, and summaries |
| `src/validation.py` | Input and dataset validation |
| `models/model_metadata.json` | Training provenance, parameters, metrics, and hashes |
| `tests/` | Unit, integration, artifact, and UI checks |

## Security and limitations

- The app uses synthetic IDs and rejects commonly named sensitive payment fields,
  but it is not a hardened data-loss-prevention system.
- SQLite queries are parameterized, and HIGH alerts are written atomically with
  their transaction records.
- Joblib artifacts use pickle semantics. Load them only from a trusted source.
- The local app has no authentication, role-based access, encryption at rest, or
  payment-gateway integration.
- The source simulates mostly US card behavior; it is not representative Indian
  payment-gateway traffic.
- Device novelty, device type, payment method, and independent merchant reputation
  are missing or unsuitable in the source and are not learned by the model.
- Results come from one synthetic dataset and one chronological split. There is no
  external validation, calibration, fairness study, or operational cost analysis.
- Manual assessments have no outcome labels, so live precision, recall, accuracy,
  and model drift cannot be measured from the local database.
- The static rules are demonstration controls, not validated fraud policy.

A real deployment would require representative data, probability calibration,
threshold tuning, outcome capture, access control, encryption, audit trails,
compliance review, monitoring, and qualified human oversight.

## Attribution

The IBM TabFormer repository and dataset carry an Apache License 2.0 notice. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the bundled
[`Apache-2.0 license text`](LICENSES/IBM-TabFormer-Apache-2.0.txt) before
redistributing upstream or derived material.

Research references:

- Erik R. Altman, [“Synthesizing Credit Card Transactions”](https://arxiv.org/abs/1910.03033)
- Inkit Padhi et al., [“Tabular Transformers for Modeling Multivariate Time Series”](https://arxiv.org/abs/2011.01843)

IBM, the TabFormer authors, Kaggle, and Razorpay do not sponsor, endorse, or
maintain AI RiskGuard. Their names are used only for dataset attribution and to
describe the example use case.
