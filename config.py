"""Central configuration for the AI RiskGuard educational prototype."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATABASE_DIR = BASE_DIR / "database"
MODELS_DIR = BASE_DIR / "models"

DATA_PATH = DATA_DIR / "transactions.csv"
SAMPLE_TRANSACTIONS_PATH = DATA_DIR / "sample_transactions.json"
DATABASE_PATH = DATABASE_DIR / "riskguard.db"
MODEL_PATH = MODELS_DIR / "fraud_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

RANDOM_STATE = 42
DATASET_SIZE = 12_000
TEST_SIZE = 0.20

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

MODEL_FEATURES = RAW_FEATURES + DERIVED_FEATURES
CATEGORICAL_FEATURES = ["payment_method", "device_type"]
NUMERICAL_FEATURES = [
    feature for feature in MODEL_FEATURES if feature not in CATEGORICAL_FEATURES
]

REQUIRED_DATASET_COLUMNS = [
    "transaction_id",
    "user_id",
    "merchant_id",
    *RAW_FEATURES,
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
    "payment risk use case. It uses synthetic/anonymized data, is not an official "
    "Razorpay product, and is not affiliated with or endorsed by Razorpay."
)


def ensure_directories() -> None:
    """Create runtime directories without modifying existing content."""
    for directory in (DATA_DIR, DATABASE_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
