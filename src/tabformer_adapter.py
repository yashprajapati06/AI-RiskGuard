"""Prepare IBM TabFormer data for model training.

TabFormer is synthetic and dollar-denominated. The adapter drops card details,
uses the project's fixed demo multiplier, and never uses the fraud label for
features or sampling. Transactions must stay grouped by user so history works
across CSV chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import zipfile
from collections import deque
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from config import (
    DATA_PATH,
    DATASET_METADATA_PATH,
    TABFORMER_DATASET_URL,
    TABFORMER_DOWNLOAD_URL,
    TABFORMER_USD_TO_INR_NORMALIZATION,
)
from src.utils import write_json

USD_TO_INR_NORMALIZATION: Final[float] = TABFORMER_USD_TO_INR_NORMALIZATION
DEFAULT_CHUNKSIZE: Final[int] = 200_000
DEFAULT_TARGET_ROWS: Final[int] = 500_000

OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "transaction_id",
    "user_id",
    "merchant_id",
    "event_timestamp",
    "amount",
    "previous_failed_txns",
    "txn_count_10min",
    "avg_user_transaction_amount",
    "location_change",
    "account_age_days",
    "hour_of_day",
    "is_weekend",
    "international_transaction",
    "fraud",
)

_INTERNAL_COLUMNS: Final[tuple[str, ...]] = ("_source_order", "_sample_key")

_TRANSACTION_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "user": ("User", "user", "user_id", "client_id", "client id"),
    "card": ("Card", "card", "card_id", "card id", "card_index"),
    "transaction_id": ("transaction_id", "transaction id", "txn_id", "id"),
    "timestamp": (
        "event_timestamp",
        "transaction_timestamp",
        "timestamp",
        "datetime",
        "date",
    ),
    "year": ("Year", "year"),
    "month": ("Month", "month"),
    "day": ("Day", "day"),
    "time": ("Time", "time"),
    "amount": ("Amount", "amount"),
    "merchant_id": (
        "merchant_id",
        "merchant id",
        "Merchant Name",
        "merchant name",
    ),
    "merchant_city": ("Merchant City", "merchant_city", "merchant city"),
    "merchant_state": ("Merchant State", "merchant_state", "merchant state"),
    "errors": ("Errors?", "errors", "error"),
    "fraud": ("Is Fraud?", "is_fraud", "is fraud", "fraud"),
}

_CARD_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "user": ("User", "user", "user_id", "client_id", "client id"),
    "card": (
        "CARD INDEX",
        "card_index",
        "Card",
        "card",
        "card_id",
        "card id",
        "id",
    ),
    "account_open_date": (
        "Acct Open Date",
        "acct_open_date",
        "account_open_date",
        "account open date",
    ),
}

_TRUE_VALUES: Final[frozenset[str]] = frozenset(
    {"1", "true", "t", "yes", "y", "fraud", "fraudulent"}
)
_FALSE_VALUES: Final[frozenset[str]] = frozenset(
    {"0", "false", "f", "no", "n", "legitimate", "non-fraud", "nonfraud"}
)
_NO_ERROR_VALUES: Final[frozenset[str]] = frozenset(
    {"", "0", "false", "n", "no", "nan", "none", "null", "no error", "no errors"}
)

_US_STATE_CODES: Final[frozenset[str]] = frozenset(
    {
        "AL",
        "AK",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "FL",
        "GA",
        "HI",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
        "DC",
        "AS",
        "GU",
        "MP",
        "PR",
        "UM",
        "VI",
    }
)
_US_STATE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "ALABAMA",
        "ALASKA",
        "ARIZONA",
        "ARKANSAS",
        "CALIFORNIA",
        "COLORADO",
        "CONNECTICUT",
        "DELAWARE",
        "FLORIDA",
        "GEORGIA",
        "HAWAII",
        "IDAHO",
        "ILLINOIS",
        "INDIANA",
        "IOWA",
        "KANSAS",
        "KENTUCKY",
        "LOUISIANA",
        "MAINE",
        "MARYLAND",
        "MASSACHUSETTS",
        "MICHIGAN",
        "MINNESOTA",
        "MISSISSIPPI",
        "MISSOURI",
        "MONTANA",
        "NEBRASKA",
        "NEVADA",
        "NEW HAMPSHIRE",
        "NEW JERSEY",
        "NEW MEXICO",
        "NEW YORK",
        "NORTH CAROLINA",
        "NORTH DAKOTA",
        "OHIO",
        "OKLAHOMA",
        "OREGON",
        "PENNSYLVANIA",
        "RHODE ISLAND",
        "SOUTH CAROLINA",
        "SOUTH DAKOTA",
        "TENNESSEE",
        "TEXAS",
        "UTAH",
        "VERMONT",
        "VIRGINIA",
        "WASHINGTON",
        "WEST VIRGINIA",
        "WISCONSIN",
        "WYOMING",
        "DISTRICT OF COLUMBIA",
        "AMERICAN SAMOA",
        "GUAM",
        "NORTHERN MARIANA ISLANDS",
        "PUERTO RICO",
        "U.S. VIRGIN ISLANDS",
        "UNITED STATES",
        "USA",
        "US",
    }
)

_TEN_MINUTES_NS: Final[int] = 10 * 60 * 1_000_000_000
_TWENTY_FOUR_HOURS_NS: Final[int] = 24 * 60 * 60 * 1_000_000_000


def _normalise_header(value: object) -> str:
    """Normalize a header for column matching."""
    return "".join(
        character for character in str(value).casefold() if character.isalnum()
    )


def _find_column(columns: Sequence[str], aliases: Iterable[str]) -> str | None:
    normalised = {_normalise_header(column): column for column in columns}
    for alias in aliases:
        match = normalised.get(_normalise_header(alias))
        if match is not None:
            return match
    return None


def _resolve_columns(
    columns: Sequence[str],
    aliases: dict[str, tuple[str, ...]],
) -> dict[str, str | None]:
    return {name: _find_column(columns, choices) for name, choices in aliases.items()}


def _validate_transaction_schema(mapping: dict[str, str | None]) -> bool:
    core_present = all(
        mapping[name] is not None
        for name in (
            "user",
            "card",
            "amount",
            "merchant_id",
            "merchant_city",
            "merchant_state",
            "errors",
            "fraud",
        )
    )
    timestamp_present = mapping["timestamp"] is not None or all(
        mapping[name] is not None for name in ("year", "month", "day", "time")
    )
    return core_present and timestamp_present


def _inspect_csv_members(
    archive: zipfile.ZipFile,
) -> tuple[str, dict[str, str | None], str, dict[str, str | None]]:
    transaction_candidates: list[tuple[int, int, str, dict[str, str | None]]] = []
    card_candidates: list[tuple[int, int, str, dict[str, str | None]]] = []

    for info in archive.infolist():
        if info.is_dir() or not info.filename.casefold().endswith(".csv"):
            continue
        try:
            with archive.open(info) as source:
                columns = list(pd.read_csv(source, nrows=0).columns)
        except (OSError, UnicodeError, pd.errors.ParserError):
            continue

        transaction_mapping = _resolve_columns(columns, _TRANSACTION_ALIASES)
        if _validate_transaction_schema(transaction_mapping):
            name_score = (
                1 if "transaction" in Path(info.filename).name.casefold() else 0
            )
            transaction_candidates.append(
                (name_score, info.file_size, info.filename, transaction_mapping)
            )

        card_mapping = _resolve_columns(columns, _CARD_ALIASES)
        if all(card_mapping[name] is not None for name in _CARD_ALIASES):
            base_name = Path(info.filename).name.casefold()
            name_score = (
                1 if "card" in base_name and "transaction" not in base_name else 0
            )
            card_candidates.append(
                (name_score, info.file_size, info.filename, card_mapping)
            )

    if not transaction_candidates:
        raise ValueError(
            "No compatible TabFormer transaction CSV was found in the ZIP archive."
        )
    if not card_candidates:
        raise ValueError(
            "No compatible TabFormer cards CSV with an account-open date was found "
            "in the ZIP archive."
        )

    transaction = max(transaction_candidates, key=lambda item: (item[0], item[1]))
    cards = max(card_candidates, key=lambda item: (item[0], item[1]))
    if transaction[2] == cards[2]:
        raise ValueError(
            "The transaction and cards CSV members could not be distinguished."
        )
    return transaction[2], transaction[3], cards[2], cards[3]


def _normalise_identifier(values: pd.Series) -> pd.Series:
    result = values.astype("string").str.strip()
    # CSV parsing can turn integer IDs into strings such as "1.0".
    return result.str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)


def _parse_datetime(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
    return parsed.dt.tz_convert(None)


def _parse_account_open_dates(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.strip()
    parsed = pd.to_datetime(cleaned, format="%m/%Y", errors="coerce", utc=True)
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            cleaned.loc[missing], errors="coerce", utc=True, format="mixed"
        )
    return parsed.dt.tz_convert(None)


def _load_card_open_dates(
    archive: zipfile.ZipFile,
    member: str,
    mapping: dict[str, str | None],
) -> dict[str, pd.Timestamp]:
    actual_columns = [mapping[name] for name in _CARD_ALIASES]
    # The archive scan normally catches this, but direct calls may not.
    if any(column is None for column in actual_columns):
        raise ValueError("The cards CSV schema is incomplete.")

    with archive.open(member) as source:
        cards = pd.read_csv(source, usecols=actual_columns, dtype="string")
    cards = cards.rename(
        columns={
            mapping["user"]: "_user",
            mapping["card"]: "_card",
            mapping["account_open_date"]: "_account_open_date",
        }
    )
    cards["_user"] = _normalise_identifier(cards["_user"])
    cards["_card"] = _normalise_identifier(cards["_card"])
    cards["_open_timestamp"] = _parse_account_open_dates(cards["_account_open_date"])

    invalid = (
        cards["_user"].isna()
        | cards["_user"].eq("")
        | cards["_card"].isna()
        | cards["_card"].eq("")
        | cards["_open_timestamp"].isna()
    )
    if invalid.any():
        raise ValueError(
            f"The cards CSV contains {int(invalid.sum())} rows with an invalid key "
            "or account-open date."
        )

    cards["_join_key"] = cards["_user"] + "\x1f" + cards["_card"]
    conflicting = cards.groupby("_join_key", sort=False)["_open_timestamp"].nunique()
    if (conflicting > 1).any():
        raise ValueError("The cards CSV contains conflicting account-open dates.")

    unique_cards = cards.drop_duplicates("_join_key", keep="first")
    return dict(zip(unique_cards["_join_key"], unique_cards["_open_timestamp"]))


def _parse_fraud(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").fillna("").str.strip().str.casefold()
    unknown = ~(cleaned.isin(_TRUE_VALUES) | cleaned.isin(_FALSE_VALUES))
    if unknown.any():
        examples = ", ".join(repr(value) for value in cleaned.loc[unknown].unique()[:3])
        raise ValueError(f"Unrecognized fraud label(s): {examples}.")
    return cleaned.isin(_TRUE_VALUES).astype("int8")


def _parse_amount(values: pd.Series) -> pd.Series:
    cleaned = (
        values.astype("string")
        .str.strip()
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    amounts_usd = pd.to_numeric(cleaned, errors="coerce")
    invalid = amounts_usd.isna() | ~np.isfinite(amounts_usd)
    if invalid.any():
        raise ValueError(
            f"The transaction CSV contains {int(invalid.sum())} invalid amounts."
        )
    return (amounts_usd.astype(float) * USD_TO_INR_NORMALIZATION).round(2)


def _parse_timestamp(
    frame: pd.DataFrame,
    mapping: dict[str, str | None],
) -> pd.Series:
    if mapping["timestamp"] is not None:
        return _parse_datetime(frame[mapping["timestamp"]])

    components = {
        part: frame[mapping[part]].astype("string").str.strip()
        for part in ("year", "month", "day", "time")
    }
    combined = (
        components["year"]
        + "-"
        + components["month"]
        + "-"
        + components["day"]
        + " "
        + components["time"]
    )
    return _parse_datetime(combined)


def _has_error(values: pd.Series) -> np.ndarray:
    cleaned = values.astype("string").fillna("").str.strip().str.casefold()
    return (~cleaned.isin(_NO_ERROR_VALUES)).to_numpy(dtype=bool)


def _rolling_history(
    timestamps: pd.Series,
    error_flags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build failure history and the rolling transaction count."""
    timestamp_ns = timestamps.astype("int64", copy=False).to_numpy()
    previous_failures = np.empty(len(timestamp_ns), dtype=np.int16)
    velocity = np.empty(len(timestamp_ns), dtype=np.int16)
    failure_times: deque[int] = deque()
    transaction_times: deque[int] = deque()

    for index, current_time in enumerate(timestamp_ns):
        failure_cutoff = current_time - _TWENTY_FOUR_HOURS_NS
        while failure_times and failure_times[0] < failure_cutoff:
            failure_times.popleft()
        previous_failures[index] = min(len(failure_times), 12)
        if error_flags[index]:
            failure_times.append(int(current_time))

        velocity_cutoff = current_time - _TEN_MINUTES_NS
        while transaction_times and transaction_times[0] < velocity_cutoff:
            transaction_times.popleft()
        transaction_times.append(int(current_time))
        velocity[index] = min(len(transaction_times), 20)

    return previous_failures, velocity


def _is_international(states: pd.Series) -> pd.Series:
    cleaned = states.astype("string").fillna("").str.strip().str.upper()
    known_us = cleaned.isin(_US_STATE_CODES | _US_STATE_NAMES)
    return (cleaned.ne("") & ~known_us).astype("int8")


def _derive_user_features(
    group: pd.DataFrame,
    mapping: dict[str, str | None],
    card_open_dates: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    group = group.copy()
    group["_timestamp"] = _parse_timestamp(group, mapping)
    if group["_timestamp"].isna().any():
        raise ValueError(
            "The transaction CSV contains an invalid event date or time for user "
            f"{group['_user'].iloc[0]!r}."
        )
    group = group.sort_values(["_timestamp", "_source_order"], kind="mergesort")

    amount = _parse_amount(group[mapping["amount"]])
    # The app handles purchases only. Drop refunds and zero-value rows before
    # building history so they cannot affect derived features.
    positive_payment = amount.gt(0)
    group = group.loc[positive_payment].copy()
    amount = amount.loc[positive_payment]
    if group.empty:
        return pd.DataFrame(columns=(*OUTPUT_COLUMNS, "_source_order"))

    errors = _has_error(group[mapping["errors"]])
    previous_failures, velocity = _rolling_history(group["_timestamp"], errors)

    running_total = amount.cumsum().shift(1)
    prior_count = pd.Series(np.arange(len(group)), index=group.index, dtype=float)
    prior_average = running_total.div(prior_count.replace(0.0, np.nan)).fillna(amount)

    city = group[mapping["merchant_city"]].astype("string").fillna("").str.strip()
    state = group[mapping["merchant_state"]].astype("string").fillna("").str.strip()
    location = city.str.casefold() + "\x1f" + state.str.casefold()
    location_known = city.ne("") | state.ne("")
    previous_known = location_known.shift(1, fill_value=False)
    location_change = (
        location_known & previous_known & location.ne(location.shift(1))
    ).astype("int8")

    card_key = group["_user"] + "\x1f" + group["_card"]
    open_timestamp = card_key.map(card_open_dates)
    if open_timestamp.isna().any():
        missing_key = card_key.loc[open_timestamp.isna()].iloc[0]
        readable_key = missing_key.replace("\x1f", "/")
        raise ValueError(
            f"No cards CSV match was found for user/card {readable_key!r}."
        )
    account_age_days = (
        (group["_timestamp"].dt.normalize() - open_timestamp.dt.normalize()).dt.days
    ).clip(lower=0)

    transaction_column = mapping["transaction_id"]
    if transaction_column is None:
        transaction_id = "tabformer-" + group["_source_order"].astype(str)
    else:
        transaction_id = _normalise_identifier(group[transaction_column])
        missing_transaction_id = transaction_id.isna() | transaction_id.eq("")
        if missing_transaction_id.any():
            transaction_id.loc[missing_transaction_id] = "tabformer-" + group.loc[
                missing_transaction_id, "_source_order"
            ].astype(str)

    merchant_id = _normalise_identifier(group[mapping["merchant_id"]])
    if merchant_id.isna().any() or merchant_id.eq("").any():
        raise ValueError("The transaction CSV contains a missing merchant identifier.")

    output = pd.DataFrame(
        {
            "transaction_id": transaction_id.astype(str),
            "user_id": group["_user"].astype(str),
            "merchant_id": merchant_id.astype(str),
            "event_timestamp": group["_timestamp"],
            "amount": amount.astype(float),
            "previous_failed_txns": previous_failures,
            "txn_count_10min": velocity,
            "avg_user_transaction_amount": prior_average.round(2).astype(float),
            "location_change": location_change,
            "account_age_days": account_age_days.astype("int32"),
            "hour_of_day": group["_timestamp"].dt.hour.astype("int8"),
            "is_weekend": (group["_timestamp"].dt.dayofweek >= 5).astype("int8"),
            "international_transaction": _is_international(state),
            "fraud": _parse_fraud(group[mapping["fraud"]]),
            "_source_order": group["_source_order"].astype("int64"),
        },
        index=group.index,
    )
    return output.reset_index(drop=True)


def _sample_keys(source_order: pd.Series, seed: int) -> np.ndarray:
    """Build stable sampling priorities from source row order."""
    values = source_order.to_numpy(dtype=np.uint64, copy=True)
    seed_value = np.uint64(seed % (1 << 64))
    with np.errstate(over="ignore"):
        values ^= seed_value
        values += np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        values ^= values >> np.uint64(31)
    return values


class _UniformCollector:
    """Keep a bottom-k sample without looking at labels."""

    def __init__(self, target_rows: int | None, seed: int) -> None:
        self.target_rows = target_rows
        self.seed = seed
        self._retained: pd.DataFrame | None = None
        self._pending: list[pd.DataFrame] = []
        self._pending_rows = 0

    def add(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        frame = frame.copy()
        if self.target_rows is not None:
            frame["_sample_key"] = _sample_keys(frame["_source_order"], self.seed)
        self._pending.append(frame)
        self._pending_rows += len(frame)

        threshold = (
            100_000 if self.target_rows is None else max(100_000, self.target_rows)
        )
        if self._pending_rows >= threshold:
            self._compact()

    def _compact(self) -> None:
        if not self._pending:
            return
        frames = (
            [self._retained] if self._retained is not None else []
        ) + self._pending
        combined = pd.concat(frames, ignore_index=True)
        self._pending = []
        self._pending_rows = 0

        if self.target_rows is not None and len(combined) > self.target_rows:
            # Source order breaks the unlikely tie between equal sample keys.
            combined = combined.nsmallest(
                self.target_rows,
                ["_sample_key", "_source_order"],
                keep="first",
            )
        self._retained = combined

    def finish(self) -> pd.DataFrame:
        self._compact()
        if self._retained is None:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        result = self._retained.sort_values(
            ["event_timestamp", "_source_order"], kind="mergesort"
        )
        result = result.drop(columns=list(_INTERNAL_COLUMNS), errors="ignore")
        return result.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)


def _validate_contiguous_users(frame: pd.DataFrame, completed_users: set[str]) -> None:
    if frame.empty:
        return
    run_start = frame["_user"].ne(frame["_user"].shift()).fillna(True)
    run_users = frame.loc[run_start, "_user"]
    duplicated_runs = run_users[run_users.duplicated()].unique()
    if len(duplicated_runs):
        raise ValueError(
            "The transaction CSV must be grouped contiguously by user; repeated "
            f"group found for {duplicated_runs[0]!r}."
        )
    repeated_completed = next(
        (user for user in run_users.astype(str) if user in completed_users), None
    )
    if repeated_completed is not None:
        raise ValueError(
            "The transaction CSV must be grouped contiguously by user; user "
            f"{repeated_completed!r} reappeared after its group ended."
        )


def _atomic_write_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        dataframe.to_csv(temporary_path, index=False, encoding="utf-8")
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _file_sha256(path: Path) -> str:
    """Hash a source file in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _console_safe_path(path: Path) -> str:
    """Format a path safely for older Windows consoles."""
    try:
        display = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        display = str(path.resolve())
    encoding = sys.stdout.encoding or "utf-8"
    return display.encode(encoding, errors="backslashreplace").decode(encoding)


def _write_training_provenance(
    output_path: Path,
    *,
    source_rows: int,
    eligible_rows: int,
    sample_rows: int,
    target_rows: int | None,
    seed: int,
) -> None:
    """Write the source manifest used during training."""
    sampling = (
        "all_positive_purchase_rows_after_label_independent_filter"
        if target_rows is None or sample_rows == eligible_rows
        else "deterministic_uniform_label_independent_bottom_k"
    )
    write_json(
        DATASET_METADATA_PATH,
        {
            "manifest_version": 1,
            "source_id": "ibm_tabformer_credit_card_transactions",
            "source_name": "IBM TabFormer fully synthetic credit-card transactions",
            "source_url": TABFORMER_DATASET_URL,
            "download_url": TABFORMER_DOWNLOAD_URL,
            "upstream_license": "Apache-2.0",
            "data_origin": "fully_synthetic_ibm_tabformer",
            "source_rows": source_rows,
            "eligible_positive_purchase_rows": eligible_rows,
            "excluded_non_positive_rows": source_rows - eligible_rows,
            "sample_rows": sample_rows,
            "target_rows": target_rows,
            "sampling_seed": seed,
            "sampling_strategy": sampling,
            "sampling_uses_target": False,
            "historical_feature_policy": (
                "Features use only the current positive payment and strictly prior "
                "positive payments for the same synthetic user; the current fraud "
                "label and future events are never used."
            ),
            "amount_filter": (
                "Source rows with Amount <= 0 are excluded as refunds, credits, or "
                "zero-value events before history calculation and sampling."
            ),
            "amount_normalization": (
                f"Positive source USD values multiplied by fixed "
                f"{USD_TO_INR_NORMALIZATION:g} for INR UI compatibility; this is "
                "not a live exchange rate."
            ),
            "dataset_sha256": _file_sha256(output_path),
        },
    )


def adapt_tabformer_archive(
    zip_path: str | Path,
    output_path: str | Path | None = None,
    *,
    chunksize: int = DEFAULT_CHUNKSIZE,
    target_rows: int | None = DEFAULT_TARGET_ROWS,
    seed: int = 42,
) -> pd.DataFrame:
    """Convert a TabFormer ZIP into training rows.

    Sampling is deterministic and does not use the fraud label. Pass
    ``target_rows=None`` to keep every eligible purchase. An output file is
    replaced only after the full conversion succeeds.
    """
    archive_path = Path(zip_path).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"TabFormer ZIP archive not found: {archive_path}")
    if isinstance(chunksize, bool) or not isinstance(chunksize, int) or chunksize <= 0:
        raise ValueError("chunksize must be a positive integer.")
    if target_rows is not None and (
        isinstance(target_rows, bool)
        or not isinstance(target_rows, int)
        or target_rows <= 0
    ):
        raise ValueError("target_rows must be a positive integer or None.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    destination = Path(output_path).expanduser().resolve() if output_path else None
    if destination is not None and destination == archive_path:
        raise ValueError("output_path must not overwrite the source ZIP archive.")

    collector = _UniformCollector(target_rows=target_rows, seed=seed)
    completed_users: set[str] = set()
    pending = pd.DataFrame()
    source_offset = 0
    eligible_rows = 0

    try:
        archive_context = zipfile.ZipFile(archive_path)
    except zipfile.BadZipFile as exc:
        raise ValueError(
            "The source must be the Kaggle .zip archive, not the repository .tgz "
            "or a Git LFS pointer."
        ) from exc

    with archive_context as archive:
        transaction_member, transaction_mapping, card_member, card_mapping = (
            _inspect_csv_members(archive)
        )
        card_open_dates = _load_card_open_dates(archive, card_member, card_mapping)

        selected_columns = list(
            dict.fromkeys(
                column for column in transaction_mapping.values() if column is not None
            )
        )
        with archive.open(transaction_member) as source:
            chunks = pd.read_csv(
                source,
                usecols=selected_columns,
                dtype="string",
                chunksize=chunksize,
                low_memory=False,
            )
            for chunk in chunks:
                if chunk.empty:
                    continue
                chunk = chunk.copy()
                chunk["_source_order"] = np.arange(
                    source_offset, source_offset + len(chunk), dtype=np.int64
                )
                source_offset += len(chunk)
                chunk["_user"] = _normalise_identifier(
                    chunk[transaction_mapping["user"]]
                )
                chunk["_card"] = _normalise_identifier(
                    chunk[transaction_mapping["card"]]
                )
                invalid_key = (
                    chunk["_user"].isna()
                    | chunk["_user"].eq("")
                    | chunk["_card"].isna()
                    | chunk["_card"].eq("")
                )
                if invalid_key.any():
                    raise ValueError(
                        "The transaction CSV contains a missing user or card "
                        "identifier."
                    )

                combined = (
                    pd.concat([pending, chunk], ignore_index=True)
                    if not pending.empty
                    else chunk.reset_index(drop=True)
                )
                _validate_contiguous_users(combined, completed_users)
                user_run = (
                    combined["_user"]
                    .ne(combined["_user"].shift())
                    .fillna(True)
                    .cumsum()
                )
                trailing_mask = user_run.eq(user_run.iloc[-1])
                pending = combined.loc[trailing_mask].copy()
                complete = combined.loc[~trailing_mask]

                for user, group in complete.groupby("_user", sort=False, observed=True):
                    derived = _derive_user_features(
                        group, transaction_mapping, card_open_dates
                    )
                    eligible_rows += len(derived)
                    collector.add(derived)
                    completed_users.add(str(user))

        if not pending.empty:
            user = str(pending["_user"].iloc[0])
            if user in completed_users:
                raise ValueError(
                    f"User {user!r} reappeared after its transaction group ended."
                )
            derived = _derive_user_features(
                pending, transaction_mapping, card_open_dates
            )
            eligible_rows += len(derived)
            collector.add(derived)

    result = collector.finish()
    if result.empty:
        raise ValueError("The transaction CSV contains no data rows.")
    if destination is not None:
        _atomic_write_csv(result, destination)
        if destination == DATA_PATH.resolve():
            _write_training_provenance(
                destination,
                source_rows=source_offset,
                eligible_rows=eligible_rows,
                sample_rows=len(result),
                target_rows=target_rows,
                seed=seed,
            )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an IBM TabFormer Kaggle ZIP into privacy-minimized, "
            "leakage-safe AI RiskGuard features."
        )
    )
    parser.add_argument(
        "zip_path", type=Path, help="Path to the downloaded ZIP archive."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_PATH,
        help="Destination CSV path (default: %(default)s).",
    )
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument(
        "--target-rows",
        type=int,
        default=DEFAULT_TARGET_ROWS,
        help="Uniform sample size (default: %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the adapter from the command line."""
    arguments = _build_parser().parse_args(argv)
    result = adapt_tabformer_archive(
        arguments.zip_path,
        arguments.output,
        chunksize=arguments.chunksize,
        target_rows=arguments.target_rows,
        seed=arguments.seed,
    )
    fraud_count = int(result["fraud"].sum())
    print(
        f"Wrote {len(result):,} rows ({fraud_count:,} fraud) to "
        f"{_console_safe_path(arguments.output)}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised manually through the CLI
    raise SystemExit(main())
