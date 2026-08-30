# AI RiskGuard

**Intelligent Payment Risk Assessment and Fraud Monitoring System**

**[Launch the public Streamlit demo](https://ai-riskguard-payment.streamlit.app/)**

AI RiskGuard is a complete local internship project for a **Razorpay-style
payment risk use case**. It demonstrates synthetic payment data engineering,
imbalanced classification, explainable prototype risk rules, bounded risk
scoring, SQLite monitoring, testing, and a professional Streamlit dashboard.

> **Disclaimer:** This is an educational prototype using synthetic/anonymized
> transaction data. It is not an official Razorpay product, is not affiliated
> with Razorpay, and must not be used for real financial decision-making or
> payment authorization.

No Razorpay logo, API credential, proprietary rule, or production asset is used.

## Submission at a glance

| Item | Implementation |
|---|---|
| Use case | Digital payment risk management prototype |
| Data | 12,000 reproducible synthetic/anonymized transactions |
| Candidate models | Logistic Regression and Random Forest |
| Selection focus | Recall, F1, ROC-AUC, and precision—not accuracy alone |
| Explainability | Triggered rule factors plus global model importance |
| Risk output | Fraud probability, ML score, rule score, final score, level, action |
| Persistence | Local SQLite transactions and HIGH-risk alerts |
| Interface | Five-page Streamlit dashboard |
| Verification | Automated tests, linting, syntax checks, and page smoke tests |

### Quick demonstration

```bash
python -m pip install -r requirements.txt
python -m src.train_model
python -m streamlit run app.py
```

Then open **Transaction Risk Checker**, analyze the Normal, Medium, and High
presets, and review the Dashboard, Monitor, Alerts, and Model Performance pages.

### Documentation guide

- [System architecture](#5-system-architecture)
- [Dataset and features](#8-dataset-description)
- [Model training and evaluation](#10-fraud-detection-models)
- [Risk scoring and rules](#13-risk-scoring-formula)
- [Installation and commands](#15-installation)
- [Security, limitations, and future scope](#22-security-and-privacy)

## 1. Project overview

Modern payment systems must distinguish a small number of risky transactions
from a much larger genuine population. RiskGuard demonstrates a clear local
architecture for this Razorpay-style payment risk use case: it accepts behavioral
transaction features, validates them, applies the exact feature engineering used
during training, obtains an actual `predict_proba()` fraud estimate, evaluates
transparent educational rules, combines both scores, stores the result, and
creates a manual-review alert for HIGH risk.

No paid API, cloud service, payment gateway credential, or real banking data is
required.

## 2. Problem statement

Payment fraud is an imbalanced classification problem: most transactions are
genuine, but missed fraud can be costly. A usable assessment also needs
understandable warning indicators, monitoring records, and honest evaluation—not
only a model prediction or a misleading accuracy number.

## 3. Objective

For every submitted synthetic transaction, the project returns:

- Fraud probability from the selected machine-learning classifier
- ML risk score (`probability × 100`)
- Normalized rule-based score
- Weighted final score from 0 to 100
- LOW, MEDIUM, or HIGH level
- Triggered prototype risk factors and human-readable reasons
- Recommended prototype action
- SQLite monitoring record and an automatic HIGH-risk alert

## 4. Main features

- Reproducible 12,000-row synthetic dataset with noisy probabilistic labels
- Approximately 2–8% fraud, with both target classes present
- Shared training/inference feature engineering, including safe `amount_ratio`
- Median imputation, standard scaling, and unknown-safe one-hot encoding
- Locked stratified test split plus five-fold training-only cross-validation
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
- How a rule engine complements an ML probability with understandable factors
- How backend services, automated tests, SQLite, and Streamlit fit together
- Why privacy, human review, honest limitations, and responsible scope matter

## 5. System architecture

```text
Synthetic/manual transaction
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
high-cardinality patterns. The fraud target is never present in inference data.

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
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
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

The dataset, model artifacts, metadata, and initial empty database are included
as reproducible demonstration artifacts and are intentionally not ignored by
Git. Running training replaces the generated model/metadata with newly computed
local results. Transaction records entered after cloning are local runtime data.

## 8. Dataset description

`src/data_generator.py` uses random seed 42 and creates 12,000 transactions when
`data/transactions.csv` is absent. Labels are sampled from a noisy nonlinear
probability based on multiple factors: amount deviation, device novelty, failed
attempts, velocity, location change, merchant risk, international status,
unusual hours, interactions, and random noise. A transaction with a risk factor
is therefore not automatically fraud, and labels cannot be reconstructed with
one deterministic rule.

| Field | Meaning |
|---|---|
| `transaction_id` | Unique anonymous transaction ID (not a model feature) |
| `user_id` | Anonymous synthetic user ID (not a model feature) |
| `merchant_id` | Anonymous synthetic merchant ID (not a model feature) |
| `amount` | Synthetic transaction amount |
| `payment_method` | UPI, Card, Wallet, or NetBanking |
| `device_type` | Android, iOS, or Web |
| `is_new_device` | Binary device novelty indicator |
| `previous_failed_txns` | Recent failed transaction count |
| `txn_count_10min` | Short-window transaction velocity |
| `avg_user_transaction_amount` | Synthetic historical user average |
| `location_change` | Binary location-change indicator |
| `merchant_risk_score` | Synthetic merchant indicator in [0, 1] |
| `account_age_days` | Synthetic account age |
| `hour_of_day` | Hour from 0 to 23 |
| `is_weekend` | Binary weekend indicator |
| `international_transaction` | Binary international indicator |
| `fraud` | Sampled target: 0 genuine, 1 fraud |

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
preventing training/inference mismatch.

## 10. Fraud detection models

Two classifiers are tuned and trained:

1. **Logistic Regression** — scaled, explainable baseline with tuned regularization,
   solver, and fraud-class weight.
2. **Random Forest** — nonlinear ensemble with tuned depth, leaf size, and class
   weight.

The outer 80/20 split is stratified with random state 42. The 20% test partition
is locked and does not influence hyperparameters or model selection. Inside the
80% training partition, both candidates use five-fold stratified cross-validation
with preprocessing fitted independently inside each fold.

A configuration is eligible only when its mean cross-validation false-positive
rate is at most 20%. Eligible configurations use this documented composite:

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
- `fraud` is used only as the target and never enters the feature matrix.
- The train/test split occurs before fitting the preprocessor.
- Cross-validation preprocessing is refitted separately inside each training fold.
- Imputation, scaling, and encoding for the saved model are fitted on all training
  rows only after tuning.
- The locked test partition is used once for final reporting, not model selection.
- The fitted preprocessor is saved and reused unchanged during inference.
- Fraud probability is selected by class label `1`, not a fixed column assumption.

### Latest generated training result

The included version 1.1 artifacts were trained on 12,000 generated rows. Compared
with the previous model, training-only tuning reduced held-out false-positive rate
from 23.99% to 19.32% and improved held-out F1 from 21.96% to 23.91%. Recall changed
from 64.84% to 60.16%, an explicit trade-off for fewer false alerts. The
authoritative metrics are always loaded from `models/model_metadata.json` and
displayed by the Model Performance page. Synthetic data and an educational
baseline do not represent production-grade fraud detection.

## 11. Why accuracy alone is not enough

If fraud is only 4% of the dataset, a classifier that predicts "genuine" for
every row obtains about 96% accuracy while detecting zero fraud. Therefore:

- **Precision** asks: of predicted fraud cases, how many were actually fraud?
- **Recall** asks: of actual fraud cases, how many did the model find?
- **F1** balances precision and recall with a harmonic mean.
- **ROC-AUC** measures probability ranking across thresholds.
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

The selected classifier returns `fraud_probability`; then:

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

## 14. Rule engine

The rule engine checks configurable indicators such as very high amount, new
device, multiple failed attempts, high velocity, location change, merchant risk,
international status, amount deviation, and unusual hour. Weights and thresholds
live in `config.py`; raw points are normalized by the maximum available points.

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

## 16. Generate data and train

Training automatically generates the dataset if it is absent, validates the
schema/size/target, trains and evaluates both models, selects one, and saves all
artifacts:

```bash
python -m src.train_model
```

To explicitly generate data without training:

```bash
python -m src.data_generator
```

## 17. Run the application

```bash
python -m streamlit run app.py
```

`streamlit run app.py` also works after virtual-environment activation. On first
launch, `app.py` safely creates missing data, model artifacts, and database once.
It does not enter a rerun loop.

## 18. Run tests

```bash
python -m pytest
```

Tests do not exercise Streamlit widgets. They cover the reusable backend and an
isolated temporary SQLite database.

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

This project uses synthetic/anonymized data and stores only behavioral fields and
anonymous IDs. It must not receive or store:

- CVV or full card number
- OTP, PIN, or UPI PIN
- Bank passwords
- Full payment credentials or authentication secrets

The validator rejects commonly named sensitive fields. No secrets or API keys
are included. `.env`, Streamlit secrets, virtual environments, caches, and logs
are ignored. SQLite writes are parameterized and duplicate transaction IDs are
handled explicitly.

Additional security boundaries:

- Anonymous IDs are used only for local monitoring and are excluded from ML features.
- Scores and probability ranges are validated again before database insertion.
- HIGH alerts are created atomically with their transaction record.
- Joblib files use pickle semantics and must be loaded only from a trusted local
  project source; untrusted model artifacts can execute malicious code.
- The local demonstration has no authentication or encryption-at-rest layer and
  must not be exposed as a production financial service.
- No official Razorpay logo, proprietary rule set, or production API is included.

## 23. Important files

| File | Purpose |
|---|---|
| `config.py` | Paths, features, thresholds, weights, categories, and actions |
| `src/data_generator.py` | Reproducible noisy synthetic data generation |
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
| `tests/` | Unit and integration tests |

## 24. Limitations

- **Data realism:** synthetic data cannot fully represent real users, merchants,
  attacks, seasonality, or changing fraud strategies.
- **Model quality:** the educational baselines still need probability calibration,
  threshold tuning, temporal validation, and broader error analysis.
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
- Add temporal validation and stronger error analysis
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
model output, not a production financial decision. The application does not
authorize, decline, or block payments. Any real-world system requires qualified
human oversight and extensive validation beyond this project.
