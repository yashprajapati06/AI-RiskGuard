# AI RiskGuard

**Intelligent Payment Risk Assessment and Fraud Monitoring System**

**[Open the Streamlit demo](https://ai-riskguard-payment.streamlit.app/)**

AI RiskGuard is a complete local internship project for a **Razorpay-style
payment risk use case**. It demonstrates synthetic payment-data adaptation,
imbalanced classification, explainable prototype risk rules, bounded risk
scoring, SQLite monitoring, testing, and a professional Streamlit dashboard.

> **Disclaimer:** This educational prototype is trained on fully synthetic IBM
> TabFormer transaction data. The source is simulated—not live payment data and
> not anonymized real cardholder data. AI RiskGuard is not an official Razorpay
> or IBM product, is not affiliated with or endorsed by either company, and must
> not be used for payment authorization or real financial decisions.

No Razorpay or IBM logo, API credential, proprietary rule, or production asset
is used. “TabFormer” identifies the upstream data source; this project trains
Logistic Regression and Random Forest, not the TabFormer transformer architecture.

## Submission at a glance

| Item | Implementation |
|---|---|
| Use case | Digital payment risk management prototype |
| Data | 500,000 rows sampled from 24,386,900 fully synthetic IBM transactions |
| Candidate models | Logistic Regression and Random Forest |
| Selection focus | Recall, F1, ROC-AUC, and precision—not accuracy alone |
| Evaluation | Chronological 400,000-row train / 100,000-row locked test split |
| Tuning | Five-fold CV on 60,000 training-only rows; refit on all 400,000 |
| Explainability | Triggered rule factors plus global model importance |
| Risk output | Uncalibrated ML likelihood, ML score, rule score, final score, level, action |
| Persistence | Local SQLite transactions and HIGH-risk alerts |
| Interface | Five-page Streamlit dashboard |
| Verification | Automated tests, linting, syntax checks, and page smoke tests |

### Quick demonstration

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open **Transaction Risk Checker**, analyze the Normal, Medium, and High
presets, and review the Dashboard, Monitor, Alerts, and Model Performance pages.

### Documentation guide

- [System architecture](#5-system-architecture)
- [Dataset and features](#8-dataset-provenance-and-adaptation)
- [Model training and evaluation](#10-fraud-detection-models)
- [Risk scoring and rules](#13-risk-scoring-formula)
- [Installation and commands](#15-installation)
- [Security, limitations, and future scope](#22-security-and-privacy)

## 1. Project overview

Modern payment systems must distinguish a small number of risky transactions
from a much larger genuine population. RiskGuard demonstrates a clear local
architecture for this Razorpay-style payment risk use case: it accepts behavioral
transaction features, validates them, applies the exact feature engineering used
during training, obtains an uncalibrated `predict_proba()` likelihood estimate, evaluates
transparent educational rules, combines both scores, stores the result, and
creates a manual-review alert for HIGH risk.

No paid API, cloud service, payment gateway credential, or real banking data is
required. Reproducing the training dataset requires a separate download of the
public upstream archive; it is a static research dataset, not a live fraud API.

## 2. Problem statement

Payment fraud is an imbalanced classification problem: most transactions are
genuine, but missed fraud can be costly. A usable assessment also needs
understandable warning indicators, monitoring records, and honest evaluation—not
only a model prediction or a misleading accuracy number.

## 3. Objective

For every submitted synthetic transaction, the project returns:

- Uncalibrated fraud-likelihood estimate from the selected classifier
- ML risk score (`likelihood estimate × 100`)
- Normalized rule-based score
- Weighted final score from 0 to 100
- LOW, MEDIUM, or HIGH level
- Triggered prototype risk factors and human-readable reasons
- Recommended prototype action
- SQLite monitoring record and an automatic HIGH-risk alert

## 4. Main features

- Deterministic 500,000-row, label-independent sample from fully synthetic data
- Natural severe imbalance: 635 fraud labels out of 500,000 rows (0.127%)
- Causal history features derived only from current or earlier positive payments
- Shared training/inference feature engineering, including safe `amount_ratio`
- Median imputation and standard scaling fitted on training data only
- Locked chronological 80/20 test split plus training-only cross-validation
- Logistic Regression and Random Forest with imbalance-aware class weights
- Actual Accuracy, Precision, Recall, F1, ROC-AUC, FPR, FNR, and confusion matrix
- Automatic fraud-oriented model selection and joblib persistence
- Configurable transparent prototype rules and bounded weighted risk scoring
- Input validation with explicit privacy checks and user-friendly messages
- Transaction and alert persistence in local SQLite
- Five Streamlit pages with charts, filters, record details, and empty states
- Actual model comparison, confusion matrix, and aligned global importance
- Pytest coverage for validation, features, rules, risk scoring, prediction, and DB alerts

### Internship learning outcomes

The project is designed so a student can demonstrate and explain:

- Why payment fraud is an imbalanced classification problem
- How train-only preprocessing prevents data leakage
- Why unique identifiers and the target must be excluded from model inputs
- How Logistic Regression and Random Forest differ
- Why recall, precision, F1, and ROC-AUC matter more than accuracy alone
- How a rule engine complements an ML likelihood estimate with understandable factors
- How backend services, automated tests, SQLite, and Streamlit fit together
- Why privacy, human review, honest limitations, and responsible scope matter

## 5. System architecture

```text
TabFormer-derived/manual transaction
          |
          v
Input validation --> Shared feature engineering
                           |
                 +---------+---------+
                 |                   |
                 v                   v
          Saved ML model       Prototype rules
          predict_proba()      + explanations
                 |                   |
                 +---------+---------+
                           v
              70% ML + 30% rule score
                           |
                LOW / MEDIUM / HIGH
                           |
                  SQLite transaction
                  + HIGH alert record
                           |
                  Streamlit monitoring
```

The unique `transaction_id`, `user_id`, and `merchant_id` are retained only for
display/storage. They are excluded from model inputs to prevent meaningless
high-cardinality patterns. `event_timestamp` is used only for chronological
ordering and the outer split. The fraud target is never present in inference data.

## 6. Technology stack

- Python 3.11+
- pandas and NumPy for data engineering
- scikit-learn for preprocessing, models, and metrics
- joblib for artifact persistence
- SQLite through Python's built-in `sqlite3`
- Streamlit for the application
- Plotly for interactive charts
- pytest for automated tests

## 7. Project folder structure

```text
AI-RiskGuard/
├── .streamlit/
│   └── config.toml
├── app.py
├── config.py
├── README.md
├── THIRD_PARTY_NOTICES.md
├── LICENSES/
│   └── IBM-TabFormer-Apache-2.0.txt
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   ├── dataset_metadata.json
│   ├── sample_transactions.json
│   └── transactions.csv
├── database/
│   └── riskguard.db
├── models/
│   ├── fraud_model.pkl
│   ├── preprocessor.pkl
│   └── model_metadata.json
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Transaction_Risk_Checker.py
│   ├── 3_Transaction_Monitor.py
│   ├── 4_Risk_Alerts.py
│   └── 5_Model_Performance.py
├── src/
│   ├── __init__.py
│   ├── bootstrap.py
│   ├── database.py
│   ├── data_generator.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── monitoring.py
│   ├── predictor.py
│   ├── preprocessing.py
│   ├── risk_engine.py
│   ├── rule_engine.py
│   ├── tabformer_adapter.py
│   ├── train_model.py
│   ├── ui_helpers.py
│   ├── utils.py
│   └── validation.py
└── tests/
    ├── conftest.py
    ├── test_database.py
    ├── test_feature_engineering.py
    ├── test_prediction.py
    ├── test_risk_engine.py
    ├── test_rule_engine.py
    └── test_validation.py
```

The processed dataset, provenance manifest, model artifacts, metadata, and
initial database are included as reproducible demonstration artifacts. The
approximately 276 MB compressed archive and 2.35 GB extracted transaction CSV are excluded
from Git. Running training replaces the model and metadata with newly computed
local results. Transaction records entered after cloning are local runtime data.

## 8. Dataset provenance and adaptation

The upstream [IBM/TabFormer repository](https://github.com/IBM/TabFormer)
publishes a fully synthetic credit-card transaction dataset. The downloadable
ZIP used here is also published as the [Credit Card Transactions dataset on
Kaggle](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions).
The records simulate consumer and merchant activity; they are **not real banking
records and not anonymized real cardholder transactions**.

The source-generation methodology is described in Erik R. Altman's
[Synthesizing Credit Card Transactions](https://arxiv.org/abs/1910.03033).
TabFormer is described in [Tabular Transformers for Modeling Multivariate Time
Series](https://arxiv.org/abs/2011.01843). The upstream repository and dataset
provide an Apache License 2.0 notice. See [Third-party notices](THIRD_PARTY_NOTICES.md)
for attribution and redistribution notes.

### Processing summary

- The archive contains 24,386,900 source rows.
- `Amount <= 0` removes 1,264,896 refunds, credits, or zero-value events before
  history calculation. This positive-payment filter does not inspect `Is Fraud?`.
- The remaining 23,122,004 eligible payments are converted to causal features.
- A deterministic bottom-k sampler with seed 42 retains 500,000 rows without
  consulting or balancing the target label.
- The final sample contains 635 fraud labels (0.127%) and keeps the source's
  natural severe imbalance.
- Positive USD amounts are multiplied by the fixed educational factor `83.0`
  for INR-interface compatibility. This is not a live exchange rate.
- `data/dataset_metadata.json` records the source, filter, sampling policy,
  output SHA-256 digest, normalization, and license.

Historical features are calculated before sampling and use the same synthetic
user's current or earlier positive payments only:

| Processed field | Derivation and use |
|---|---|
| `transaction_id` | Synthetic source/generated ID; storage only |
| `user_id` | Synthetic user ID; grouping/storage only |
| `merchant_id` | Synthetic merchant ID; storage only |
| `event_timestamp` | Source date/time; chronological ordering and split only |
| `amount` | Positive source USD amount × fixed 83.0 INR normalization |
| `previous_failed_txns` | Earlier nonblank `Errors?` events in the prior 24 hours, excluding the current row; capped at 12 |
| `txn_count_10min` | Current plus earlier same-user payments in the trailing 10 minutes; capped at 20 |
| `avg_user_transaction_amount` | Expanding mean of strictly earlier payments; first-row fallback uses the current amount |
| `location_change` | Current known merchant location differs from the immediately previous known location; not device GPS |
| `account_age_days` | Transaction date minus the matched synthetic card's account-open date |
| `hour_of_day` / `is_weekend` | Derived from the event timestamp |
| `international_transaction` | Merchant state is outside recognized US states/territories; blank/unknown maps to 0 |
| `fraud` | Normalized source `Is Fraud?` label; target only |

`Errors?` is a source processing-error field, so `previous_failed_txns` is an
educational proxy rather than a confirmed failed-authentication count. Likewise,
`location_change` is a merchant-location change, not a cardholder location check.

### Model features versus manual inputs

| Input group | Effect in the included v2 model |
|---|---|
| Amount, error-history proxy, velocity, historical average, location change, account age, hour, weekend, international | Used by ML and by applicable rules |
| `is_new_device`, `merchant_risk_score` | Unavailable from the source; manual prototype inputs used only by rules |
| `payment_method`, `device_type` | Manual context/storage fields; not used by the v2 ML model or rules |
| IDs, timestamp, fraud target | Explicitly excluded from ML |

The source does not provide trustworthy equivalents for the project's device
fields or an independent merchant reputation signal, and its card interaction
modes do not match the UPI/Card/Wallet/NetBanking UI categories. These values are
not fabricated for training. In particular, `merchant_risk_score` is a manual
prototype indicator—not a platform's fraud percentage—and is never calculated
from fraud labels.

## 9. Feature engineering

The mandatory derived feature is:

```text
amount_ratio = amount / max(avg_user_transaction_amount, 0.000001)
```

The epsilon prevents division-by-zero. The pipeline also creates:

- `is_high_amount`
- `is_high_velocity`
- `failed_attempt_risk`
- `unusual_hour`

`engineer_features()` is called by both `train_model.py` and `predictor.py`,
preventing training/inference mismatch. The adapter's historical features are
computed separately because they require ordered user history; their current-row
values and names form the inference contract used by the manual checker.

## 10. Fraud detection models

Two classifiers are tuned and trained:

1. **Logistic Regression** — scaled, explainable baseline with tuned regularization,
   solver, and fraud-class weight.
2. **Random Forest** — nonlinear ensemble with tuned depth, leaf size, and class
   weight.

Rows are sorted by `event_timestamp`. The earliest 400,000 rows (January 1991 to
May 2017) form the 80% training partition; the latest 100,000 rows (May 2017 to
February 2020) are locked for final testing. The later test period cannot affect
hyperparameters or model selection.

For tractable tuning, a deterministic stratified sample of 60,000 rows is drawn
only from the training partition. Both candidates use five-fold stratified,
shuffled cross-validation on that sample, with preprocessing fitted separately
inside each fold. The chosen parameters for each candidate are then refitted on
all 400,000 training rows before the chronological test is evaluated.

A configuration is eligible only when its mean cross-validation false-positive
rate is at most 5%. Eligible configurations use this documented composite:

```text
selection score = 0.35 × recall
                + 0.35 × F1
                + 0.20 × ROC-AUC
                + 0.10 × precision
```

This emphasizes fraud recall and F1, then ROC-AUC and precision, without allowing
an extreme false-alert rate to win. The exact cross-validation parameters and
final test metrics for **both** models are saved in
`models/model_metadata.json` and shown in the Model Performance page. They are
never hardcoded or fabricated.

### Leakage controls

- `transaction_id`, `user_id`, and `merchant_id` are never model features.
- `event_timestamp` orders the split but never enters the feature matrix.
- `fraud` is used only as the target and never derives an input feature.
- Non-positive amount filtering and bottom-k sampling are label-independent.
- Historical averages, errors, velocity, and location use current/prior positive
  payments only; current error state, current label, and future rows are excluded.
- Merchant risk is never target-encoded; missing device/reputation data is not
  fabricated.
- The train/test split occurs before fitting the preprocessor.
- Cross-validation preprocessing is refitted separately inside each training fold.
- Imputation and scaling for the saved model are fitted on all training rows only
  after tuning.
- The locked test partition is used once for final reporting, not model selection.
- The fitted preprocessor is saved and reused unchanged during inference.
- The fraud-likelihood column is selected by class label `1`, not a fixed index assumption.

### Included v2 training result

Training-only cross-validation selected **Logistic Regression**. These are actual
metrics from the untouched 100,000-row chronological test partition:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR | Confusion matrix `[[TN, FP], [FN, TP]]` |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Logistic Regression | 97.53% | 4.16% | 95.54% | 7.96% | 0.9733 | 2.47% | 4.46% | `[[97420, 2468], [5, 107]]` |
| Random Forest | 95.87% | 0.86% | 31.25% | 1.67% | 0.8392 | 4.06% | 68.75% | `[[95832, 4056], [77, 35]]` |

The selected model found 107 of 112 fraud-labelled test rows, but it also flagged
2,468 genuine rows. Its low precision is therefore material and must not be hidden
behind 97.53% accuracy or high recall. The authoritative full-precision metrics,
parameters, periods, counts, and provenance are loaded from
`models/model_metadata.json`; they are not fabricated in the UI.

The classifiers use strong class weights because fraud is extremely rare. As a
result, Logistic Regression's `predict_proba()` output is **not probability
calibrated**. The application retains the name `fraud_probability` for its model
interface, but the number should be interpreted as an experimental relative-risk
score—not the empirical chance that a real payment is fraudulent. Calibration,
threshold selection, and external validation are required before any operational
interpretation.

## 11. Why accuracy alone is not enough

With only 0.127% fraud in this dataset, an always-genuine classifier would exceed
99.8% accuracy while detecting no fraud. Therefore:

- **Precision** asks: of predicted fraud cases, how many were actually fraud?
- **Recall** asks: of actual fraud cases, how many did the model find?
- **F1** balances precision and recall with a harmonic mean.
- **ROC-AUC** measures likelihood-score ranking across thresholds.
- **False Positive Rate** measures genuine transactions incorrectly flagged.
- **False Negative Rate** measures fraud transactions incorrectly considered genuine.

High false negatives can permit fraud loss. Excessive false positives can harm
genuine customers. The dashboard reports the trade-off rather than hiding it.

## 12. Confusion matrix

The matrix is computed with `confusion_matrix(y_test, y_pred, labels=[0, 1])`:

```text
[[TN, FP],
 [FN, TP]]
```

- True Positive: actual fraud correctly detected
- True Negative: genuine transaction correctly identified
- False Positive: genuine transaction incorrectly marked as fraud
- False Negative: fraud transaction incorrectly marked genuine

Rows represent actual classes and columns represent predicted classes. Therefore,
the top-left cell is TN, top-right is FP, bottom-left is FN, and bottom-right is
TP. This orientation is labelled explicitly in the Streamlit chart.

`FPR = FP / (FP + TN)` and `FNR = FN / (FN + TP)` use safe denominator checks.
ROC-AUC uses fraud probabilities rather than hard predictions.

## 13. Risk scoring formula

The selected classifier returns the interface field `fraud_probability`; then:

```text
ML risk score = fraud_probability × 100
final risk score = 0.70 × ML risk score + 0.30 × rule risk score
```

Inputs and outputs are clamped to [0, 100]. For ML=80 and rules=60:

```text
0.70 × 80 + 0.30 × 60 = 56 + 18 = 74 (HIGH)
```

| Final score | Level | Recommended prototype action |
|---|---|---|
| 0 to <35 | LOW | Approve / Normal Monitoring |
| 35 to <70 | MEDIUM | Additional Verification or Manual Review Recommended |
| 70 to 100 | HIGH | Flag for Immediate Manual Review |

The application recommends review only; it does not block a real payment.
Because the class-weighted model is not calibrated, this score is a prototype
risk-ranking input and not a literal real-world fraud probability.

## 14. Rule engine

The rule engine checks configurable indicators such as very high amount, new
device, multiple failed attempts, high velocity, location change, merchant risk,
international status, amount deviation, and unusual hour. Weights and thresholds
live in `config.py`; raw points are normalized by the maximum available points.

The TabFormer-derived ML model does not learn `is_new_device` or
`merchant_risk_score`; changing them affects only the rule component. Payment
method and device type are kept for manual context and monitoring but affect
neither the v2 ML score nor the current rules. This separation is intentional and
is shown honestly rather than filling unavailable source fields with fake values.

**These are prototype risk rules for educational purposes.** They are not real
Razorpay rules. Their human-readable messages are labelled "Triggered Risk
Factors", not an exact or causal AI decision explanation.

The reasons explain only the transparent rule component. They must not be
presented as causal explanations of every internal ML calculation. A transaction
can trigger no rule and still receive non-zero ML risk because the classifier
evaluates the complete feature pattern.

## 15. Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate on Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Activate on Linux/macOS:

```bash
source .venv/bin/activate
```

Install only the declared dependencies:

```bash
python -m pip install -r requirements.txt
```

## 16. Reproduce the dataset and train

The included artifacts can be used without downloading the large source. To
reproduce them, download the Kaggle ZIP linked in Section 8. It must contain both
the transactions and cards CSV files. Do not pass the repository `.tgz` file or a
Git LFS pointer to this adapter.

From the project root, quote Windows paths that contain spaces:

```powershell
python -m src.tabformer_adapter "C:\path\to\credit-card-transactions.zip" --target-rows 500000 --chunksize 200000 --seed 42
python -m src.train_model
python -m pytest
```

The adapter defaults to `data/transactions.csv` and atomically writes
`data/dataset_metadata.json`. It applies the positive-only filter, calculates
causal history across all eligible source rows, and performs deterministic
label-independent sampling. Training validates this provenance, makes the locked
chronological 80/20 split, tunes both candidates on training-only data, refits
each candidate on all training rows, evaluates the test period, and replaces the
saved artifacts.

The compressed source archive is approximately 276 MB and its transaction CSV is
approximately 2.35 GB uncompressed. Conversion/training can take substantial
memory and CPU time. Never commit the raw ZIP or extracted
source CSV files. `python -m src.data_generator` remains a small offline fallback
for tests or recovery; an artifact trained from it must not be described as
TabFormer-derived.

## 17. Run the application

```bash
python -m streamlit run app.py
```

`streamlit run app.py` also works after virtual-environment activation. On first
launch, `app.py` safely initializes missing local runtime resources once. The
repository includes compatible v2 model artifacts, so normal demo startup does
not retrain the 500,000-row dataset or enter a rerun loop.

## 18. Run tests

```bash
python -m pytest
```

Tests cover the reusable backend, an isolated temporary SQLite database, and the
Transaction Risk Checker form through Streamlit's `AppTest`. A deterministic QA
batch also runs 20 random held-out transactions (10 from each target class for
coverage only) through validation, class-aware probability extraction, rules,
final scoring, insertion, alert creation, and duplicate protection. The target
is never passed to inference, and this deliberately balanced smoke batch is not
reported as a model-performance estimate.

To run only the transaction checker QA:

```bash
python -m pytest tests/test_transaction_checker_qa.py -q
```

Additional verification commands:

```bash
python -m compileall -q app.py config.py src pages tests
python -c "from src.bootstrap import initialize_project; print(initialize_project())"
```

## 19. Dashboard pages

1. **Dashboard** — six KPIs, risk/method distributions, score histogram, daily
   trends, high-risk trend, and newest alerts from SQLite.
2. **Transaction Risk Checker** — normal/medium/high presets, validated form,
   actual scoring, saved result, and triggered factors.
3. **Transaction Monitor** — risk/method/score/ID filters, newest-first table,
   full details, and optional filtered CSV download.
4. **Risk Alerts** — newest HIGH-risk alerts and review explanations.
5. **Model Performance** — actual comparison, selected metrics, confusion matrix,
   educational definitions, feature importance, and monitoring summary.

Empty databases and empty filter results show helpful messages instead of errors.

## 20. Example transactions

All exact fields are in `data/sample_transactions.json` and the checker loads
them directly:

- **Normal:** ₹500, known device, no failures, normal hour, low merchant risk.
- **Medium:** ₹15,000, new device, two failures, moderately unusual amount.
- **High:** ₹80,000, new device, six failures, ten transactions in ten minutes,
  location change, 0.85 merchant risk, unusual hour, international, and 40× average.

Exact probabilities are intentionally not asserted because they come from the
trained probabilistic model. The fixed included artifacts order the three final
scores LOW < MEDIUM < HIGH.

## 21. Monitoring behavior

The live dashboard can report average prediction risk, level distribution, and
HIGH-risk count because these require no outcome label. FPR/FNR are explicitly
identified as held-out test-set metrics. The application states:

> Live accuracy cannot be calculated until ground-truth labels are available.

It never presents unlabeled manual predictions as confirmed live accuracy or
confirmed model degradation.

This distinction is important during a demonstration: the Dashboard shows
**prediction monitoring**, while the Model Performance page shows **held-out
evaluation**. They answer different questions and must not be mixed.

## 22. Security and privacy

This project uses fully synthetic source records and stores only behavioral fields
and synthetic IDs. The adapter selects only the card index and account-open date
from the source card table; it does not copy card numbers, CVV, PIN, expiry,
person names, email addresses, postal addresses, or raw merchant locations into
the processed model dataset. The application must never receive or store:

- CVV or full card number
- OTP, PIN, or UPI PIN
- Bank passwords
- Full payment credentials or authentication secrets

The validator rejects commonly named sensitive fields. No secrets or API keys
are included. `.env`, Streamlit secrets, virtual environments, caches, and logs
are ignored. Large upstream archives, extracted source CSVs, and `data/raw/` are
also ignored. SQLite writes are parameterized and duplicate transaction IDs are
handled explicitly.

Additional security boundaries:

- Synthetic IDs are used only for history grouping/local monitoring and are
  excluded from ML features.
- Scores and probability ranges are validated again before database insertion.
- HIGH alerts are created atomically with their transaction record.
- Joblib files use pickle semantics and must be loaded only from a trusted local
  project source; untrusted model artifacts can execute malicious code.
- The local demonstration has no authentication or encryption-at-rest layer and
  must not be exposed as a production financial service.
- No official Razorpay or IBM logo, proprietary rule set, or production API is
  included, and neither company endorses this prototype.

## 23. Important files

| File | Purpose |
|---|---|
| `config.py` | Paths, features, thresholds, weights, categories, and actions |
| `src/tabformer_adapter.py` | Privacy-minimized, causal, label-independent source adaptation |
| `data/dataset_metadata.json` | Source, filter, sample, checksum, currency, and license provenance |
| `src/data_generator.py` | Small reproducible synthetic fallback; not the v2 source |
| `src/validation.py` | Transaction and training-dataset validation |
| `src/feature_engineering.py` | Training/inference derived features |
| `src/preprocessing.py` | Train-only fitted `ColumnTransformer` factory |
| `src/train_model.py` | Split, train, evaluate, select, and persist |
| `src/evaluation.py` | Metrics, selection score, and global importance |
| `src/predictor.py` | Reusable `analyze_transaction()` service |
| `src/rule_engine.py` | Transparent prototype rule indicators |
| `src/risk_engine.py` | Bounded weighted score and classification |
| `src/database.py` | SQLite schema, atomic save, queries, and alerts |
| `src/monitoring.py` | Honest held-out/live monitoring distinction |
| `src/bootstrap.py` | Safe first-run initialization |
| `app.py` and `pages/` | Streamlit home and five dashboard pages |
| `models/model_metadata.json` | Actual metrics for both trained models |
| `THIRD_PARTY_NOTICES.md` | IBM dataset attribution and Apache-2.0 notice |
| `LICENSES/IBM-TabFormer-Apache-2.0.txt` | Redistributed upstream license text |
| `tests/` | Unit and integration tests |

## 24. Limitations

- **Data realism:** the source simulates mostly US card behavior. It is not Indian
  payment-gateway traffic and cannot establish real Razorpay-style performance.
- **Sampling:** the model uses a deterministic 500,000-row sample, not every one
  of the 24,386,900 source rows; sampling uncertainty remains.
- **Currency:** the fixed 83.0 USD-to-INR normalization is a reproducible UI
  assumption, not a current exchange rate or financial conversion service.
- **Feature proxies:** source errors are not confirmed authentication failures;
  merchant-location changes are not device GPS; the international heuristic can
  misclassify blank, online, or unusually formatted merchants.
- **Missing signals:** device novelty/type and independent merchant reputation are
  absent from training. Manual inputs for them must not be described as learned.
- **Model quality:** class-weighted `predict_proba()` is uncalibrated. Low held-out
  precision, threshold sensitivity, a single chronological split, and lack of
  external validation prevent probability or production claims.
- **Rule quality:** static prototype rules are understandable but are not calibrated
  production controls and may overlap with ML signals.
- **Live evaluation:** manually analyzed transactions have no ground-truth outcome,
  so live precision, recall, and accuracy are unknown.
- **Operational scope:** the app is local and single-user, with no authentication,
  encryption at rest, audit roles, or payment authorization integration.
- **Governance:** a real system requires representative data, security/compliance
  review, fairness analysis, incident handling, human oversight, and governance.

## 25. Future improvements

Future work is intentionally prioritized rather than added to the internship build:

**Near term**

- Calibrate probabilities and tune thresholds against review capacity and cost
- Add rolling temporal validation and stronger false-positive error analysis
- Capture human-review outcomes to create an honest feedback dataset

**Medium term**

- Add careful SHAP-based local explanations and distribution-shift indicators
- Compare gradient boosting and anomaly-detection approaches
- Introduce model versioning, review status, and role-based access control

**Long term**

- Study time-series, graph, device, user, and merchant behavior models
- Add a secured prediction API and real-time event processing
- Implement production-grade encryption, monitoring, compliance, and governance

## 26. Responsible-use disclaimer

AI RiskGuard is a student demonstration. Fraud probability is an experimental
uncalibrated model output, not a production financial decision or measured fraud
likelihood. The application does not authorize, decline, or block payments. Any
real-world system requires qualified human oversight and extensive validation
beyond this project. IBM, TabFormer's authors, and Razorpay do not sponsor or
endorse AI RiskGuard.
