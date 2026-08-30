"""Central configuration for the AI RiskGuard educational prototype."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATABASE_DIR = BASE_DIR / "database"
MODELS_DIR = BASE_DIR / "models"

DATA_PATH = DATA_DIR / "transactions.csv"
DATASET_METADATA_PATH = DATA_DIR / "dataset_metadata.json"
SAMPLE_TRANSACTIONS_PATH = DATA_DIR / "sample_transactions.json"
DATABASE_PATH = DATABASE_DIR / "riskguard.db"
MODEL_PATH = MODELS_DIR / "fraud_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
ARTIFACT_SCHEMA_VERSION = "2.0.0"

RANDOM_STATE = 42
DATASET_SIZE = 12_000
TEST_SIZE = 0.20
EVENT_TIMESTAMP_COLUMN = "event_timestamp"

# The included model is trained from IBM's fully synthetic TabFormer transaction
# data. The fixed multiplier only aligns the source's dollar-denominated values
# with this prototype's INR user interface; it is not a live exchange rate.
TABFORMER_DATASET_URL = "https://github.com/IBM/TabFormer"
TABFORMER_KAGGLE_HANDLE = "ealtman2019/credit-card-transactions"
TABFORMER_DOWNLOAD_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/{TABFORMER_KAGGLE_HANDLE}"
)
TABFORMER_ARCHIVE_MEMBER = "credit_card_transactions-ibm_v2.csv"
TABFORMER_CARDS_MEMBER = "sd254_cards.csv"
TABFORMER_SOURCE_ROWS = 24_386_900
TABFORMER_TARGET_ROWS = 500_000
TABFORMER_USD_TO_INR_NORMALIZATION = 83.0
CV_TUNING_MAX_ROWS = 60_000

ALLOWED_PAYMENT_METHODS = ("UPI", "Card", "Wallet", "NetBanking")
ALLOWED_DEVICE_TYPES = ("Android", "iOS", "Web")

RAW_FEATURES = [
    "amount",
    "payment_method",
    "device_type",
    "is_new_device",
    "previous_failed_txns",
    "txn_count_10min",
    "avg_user_transaction_amount",
    "location_change",
    "merchant_risk_score",
    "account_age_days",
    "hour_of_day",
    "is_weekend",
    "international_transaction",
]

DERIVED_FEATURES = [
    "amount_ratio",
    "is_high_amount",
    "is_high_velocity",
    "failed_attempt_risk",
    "unusual_hour",
]

# The IBM source does not contain trustworthy device novelty/type or an
# independently supplied merchant risk score. Payment method is constant (Card).
# Those inputs remain available as rule/context inputs, but are excluded from ML
# rather than being fabricated.
MODEL_RAW_FEATURES = [
    "amount",
    "previous_failed_txns",
    "txn_count_10min",
    "avg_user_transaction_amount",
    "location_change",
    "account_age_days",
    "hour_of_day",
    "is_weekend",
    "international_transaction",
]
NON_MODEL_INPUT_FEATURES = [
    feature for feature in RAW_FEATURES if feature not in MODEL_RAW_FEATURES
]
MODEL_FEATURES = MODEL_RAW_FEATURES + DERIVED_FEATURES
CATEGORICAL_FEATURES: list[str] = []
NUMERICAL_FEATURES = [
    feature for feature in MODEL_FEATURES if feature not in CATEGORICAL_FEATURES
]

REQUIRED_DATASET_COLUMNS = [
    "transaction_id",
    "user_id",
    "merchant_id",
    *MODEL_RAW_FEATURES,
    "fraud",
]

AMOUNT_RATIO_EPSILON = 1e-6
FEATURE_HIGH_AMOUNT_THRESHOLD = 25_000.0
FEATURE_HIGH_VELOCITY_THRESHOLD = 5
FEATURE_FAILED_ATTEMPT_THRESHOLD = 3
UNUSUAL_HOURS = (0, 1, 2, 3, 4)

# Combined score configuration.
ML_WEIGHT = 0.70
RULE_WEIGHT = 0.30
LOW_RISK_THRESHOLD = 35.0
HIGH_RISK_THRESHOLD = 70.0

# Prototype risk rules for educational purposes. These are not production rules.
RULE_THRESHOLDS = {
    "high_amount": 25_000.0,
    "failed_transactions": 3,
    "high_velocity": 5,
    "merchant_risk": 0.70,
    "amount_ratio": 3.0,
}

RULE_WEIGHTS = {
    "VERY_HIGH_AMOUNT": 20,
    "NEW_DEVICE": 10,
    "MULTIPLE_FAILED_ATTEMPTS": 15,
    "HIGH_VELOCITY": 15,
    "LOCATION_CHANGE": 10,
    "HIGH_MERCHANT_RISK": 10,
    "INTERNATIONAL_TRANSACTION": 10,
    "AMOUNT_DEVIATION": 20,
    "UNUSUAL_HOUR": 5,
}

RISK_ACTIONS = {
    "LOW": "Approve / Normal Monitoring",
    "MEDIUM": "Additional Verification or Manual Review Recommended",
    "HIGH": "Flag for Immediate Manual Review",
}

MODEL_SELECTION_WEIGHTS = {
    "recall": 0.35,
    "f1": 0.35,
    "roc_auc": 0.20,
    "precision": 0.10,
}

DISCLAIMER = (
    "Educational digital payment risk management prototype for a Razorpay-style "
    "payment risk use case. It uses fully synthetic data—not real or anonymized "
    "customer data. It is not an official Razorpay or IBM product and is not "
    "affiliated with, sponsored by, or endorsed by Razorpay, IBM, or the TabFormer "
    "authors."
)


def ensure_directories() -> None:
    """Create runtime directories without modifying existing content."""
    for directory in (DATA_DIR, DATABASE_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
