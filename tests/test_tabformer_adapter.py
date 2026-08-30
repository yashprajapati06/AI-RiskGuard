"""Tests for the privacy-preserving IBM TabFormer archive adapter."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src import tabformer_adapter
from src.tabformer_adapter import OUTPUT_COLUMNS, adapt_tabformer_archive
from src.utils import read_json


def _build_tabformer_zip(path: Path) -> None:
    cards = pd.DataFrame(
        [
            {
                "User": "1",
                "CARD INDEX": "0",
                "Card Number": "4111111111111111",
                "CVV": "123",
                "Acct Open Date": "01/2020",
            },
            {
                "User": "2",
                "CARD INDEX": "0",
                "Card Number": "5555555555554444",
                "CVV": "456",
                "Acct Open Date": "06/2019",
            },
        ]
    )
    # Users are contiguous, as in the source archive.  User 1 is deliberately
    # out of time order and spans chunks so the adapter must buffer and sort it.
    transactions = pd.DataFrame(
        [
            {
                "User": "1",
                "Card": "0",
                "Year": "2021",
                "Month": "1",
                "Day": "2",
                "Time": "10:05",
                "Amount": "$20.00",
                "Merchant Name": "merchant-a",
                "Merchant City": "New York",
                "Merchant State": "NY",
                "Errors?": "Bad PIN",
                "Is Fraud?": "No",
            },
            {
                "User": "1",
                "Card": "0",
                "Year": "2021",
                "Month": "1",
                "Day": "2",
                "Time": "10:00",
                "Amount": "$10.00",
                "Merchant Name": "merchant-a",
                "Merchant City": "New York",
                "Merchant State": "NY",
                "Errors?": "",
                "Is Fraud?": "No",
            },
            {
                "User": "1",
                "Card": "0",
                "Year": "2021",
                "Month": "1",
                "Day": "2",
                "Time": "10:02",
                "Amount": "-$5.00",
                "Merchant Name": "excluded-refund",
                "Merchant City": "Paris",
                "Merchant State": "France",
                "Errors?": "Bad PIN",
                "Is Fraud?": "Yes",
            },
            {
                "User": "1",
                "Card": "0",
                "Year": "2021",
                "Month": "1",
                "Day": "2",
                "Time": "10:04",
                "Amount": "$0.00",
                "Merchant Name": "excluded-zero",
                "Merchant City": "Paris",
                "Merchant State": "France",
                "Errors?": "Bad PIN",
                "Is Fraud?": "Yes",
            },
            {
                "User": "1",
                "Card": "0",
                "Year": "2021",
                "Month": "1",
                "Day": "2",
                "Time": "10:09",
                "Amount": "$30.00",
                "Merchant Name": "merchant-b",
                "Merchant City": "Los Angeles",
                "Merchant State": "CA",
                "Errors?": "",
                "Is Fraud?": "Yes",
            },
            {
                "User": "1",
                "Card": "0",
                "Year": "2021",
                "Month": "1",
                "Day": "2",
                "Time": "10:20",
                "Amount": "$40.00",
                "Merchant Name": "merchant-c",
                "Merchant City": "Paris",
                "Merchant State": "France",
                "Errors?": "Insufficient Balance",
                "Is Fraud?": "No",
            },
            {
                "User": "1",
                "Card": "0",
                "Year": "2021",
                "Month": "1",
                "Day": "3",
                "Time": "10:05",
                "Amount": "$50.00",
                "Merchant Name": "merchant-c",
                "Merchant City": "Paris",
                "Merchant State": "France",
                "Errors?": "",
                "Is Fraud?": "No",
            },
            {
                "User": "2",
                "Card": "0",
                "Year": "2021",
                "Month": "1",
                "Day": "2",
                "Time": "08:00",
                "Amount": "$5.00",
                "Merchant Name": "merchant-d",
                "Merchant City": "Boston",
                "Merchant State": "MA",
                "Errors?": "",
                "Is Fraud?": "Yes",
            },
            {
                "User": "2",
                "Card": "0",
                "Year": "2021",
                "Month": "1",
                "Day": "2",
                "Time": "08:03",
                "Amount": "$7.00",
                "Merchant Name": "merchant-d",
                "Merchant City": "Boston",
                "Merchant State": "MA",
                "Errors?": "",
                "Is Fraud?": "No",
            },
        ]
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("archive/sd254_cards.csv", cards.to_csv(index=False))
        archive.writestr(
            "archive/card_transaction.v1.csv", transactions.to_csv(index=False)
        )


def test_adapter_derives_chronological_history_and_excludes_private_columns(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "tabformer.zip"
    output_path = tmp_path / "processed" / "riskguard.csv"
    _build_tabformer_zip(archive_path)

    result = adapt_tabformer_archive(
        archive_path,
        output_path,
        chunksize=2,
        target_rows=None,
        seed=17,
    )

    assert tuple(result.columns) == OUTPUT_COLUMNS
    assert len(result) == 7
    assert result["amount"].gt(0).all()
    assert not result["merchant_id"].isin({"excluded-refund", "excluded-zero"}).any()
    assert output_path.is_file()
    assert not (tmp_path / "archive").exists()  # ZIP was read without extraction.
    assert not {"Card", "Card Number", "CVV", "Errors?"}.intersection(result.columns)
    assert result["event_timestamp"].is_monotonic_increasing

    user_one = result.loc[result["user_id"] == "1"].reset_index(drop=True)
    assert user_one["amount"].tolist() == [830.0, 1660.0, 2490.0, 3320.0, 4150.0]
    assert user_one["previous_failed_txns"].tolist() == [0, 0, 1, 1, 2]
    assert user_one["txn_count_10min"].tolist() == [1, 2, 3, 1, 1]
    assert user_one["avg_user_transaction_amount"].tolist() == pytest.approx(
        [830.0, 830.0, 1245.0, 1660.0, 2075.0]
    )
    assert user_one["location_change"].tolist() == [0, 0, 1, 1, 0]
    assert user_one["international_transaction"].tolist() == [0, 0, 0, 1, 1]
    assert user_one["fraud"].tolist() == [0, 0, 1, 0, 0]
    assert user_one.loc[0, "account_age_days"] == 367
    assert user_one.loc[0, "hour_of_day"] == 10
    assert user_one.loc[0, "is_weekend"] == 1

    saved = pd.read_csv(output_path)
    assert len(saved) == len(result)
    assert list(saved.columns) == list(OUTPUT_COLUMNS)


def test_uniform_sample_is_deterministic_across_chunk_sizes(tmp_path: Path) -> None:
    archive_path = tmp_path / "tabformer.zip"
    _build_tabformer_zip(archive_path)

    first = adapt_tabformer_archive(archive_path, chunksize=2, target_rows=4, seed=2026)
    second = adapt_tabformer_archive(
        archive_path, chunksize=5, target_rows=4, seed=2026
    )

    assert len(first) == 4
    pd.testing.assert_frame_equal(first, second)
    assert set(first["fraud"].unique()).issubset({0, 1})


def test_nonpositive_events_are_removed_before_history(tmp_path: Path) -> None:
    archive_path = tmp_path / "tabformer.zip"
    _build_tabformer_zip(archive_path)

    result = adapt_tabformer_archive(
        archive_path, chunksize=1, target_rows=None, seed=3
    )
    user_one = result.loc[result["user_id"] == "1"].reset_index(drop=True)

    assert result["amount"].gt(0).all()
    assert not result["merchant_id"].str.startswith("excluded-").any()
    # The removed error rows at 10:02 and 10:04 must not inflate the 10:05
    # transaction's prior-error count or 10-minute transaction velocity.
    assert user_one.loc[1, "event_timestamp"] == pd.Timestamp("2021-01-02 10:05")
    assert user_one.loc[1, "previous_failed_txns"] == 0
    assert user_one.loc[1, "txn_count_10min"] == 2
    assert user_one.loc[1, "avg_user_transaction_amount"] == 830.0


def test_adapter_rejects_non_contiguous_user_groups(tmp_path: Path) -> None:
    archive_path = tmp_path / "tabformer.zip"
    cards = pd.DataFrame(
        [
            ["1", "0", "01/2020"],
            ["2", "0", "01/2020"],
        ],
        columns=["User", "CARD INDEX", "Acct Open Date"],
    )
    interleaved = pd.DataFrame(
        [
            ["1", "0", 2021, 1, 1, "00:00", "$1", "m", "c", "NY", "", "No"],
            ["2", "0", 2021, 1, 1, "00:01", "$1", "m", "c", "NY", "", "No"],
            ["1", "0", 2021, 1, 1, "00:02", "$1", "m", "c", "NY", "", "No"],
        ],
        columns=[
            "User",
            "Card",
            "Year",
            "Month",
            "Day",
            "Time",
            "Amount",
            "Merchant Name",
            "Merchant City",
            "Merchant State",
            "Errors?",
            "Is Fraud?",
        ],
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("sd254_cards.csv", cards.to_csv(index=False))
        archive.writestr(
            "archive/transactions_interleaved.csv",
            interleaved.to_csv(index=False),
        )

    with pytest.raises(ValueError, match="grouped contiguously by user"):
        adapt_tabformer_archive(archive_path, chunksize=10, target_rows=None, seed=1)


def test_training_output_writes_verifiable_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "tabformer.zip"
    output_path = tmp_path / "data" / "transactions.csv"
    metadata_path = tmp_path / "data" / "dataset_metadata.json"
    _build_tabformer_zip(archive_path)
    monkeypatch.setattr(tabformer_adapter, "DATA_PATH", output_path)
    monkeypatch.setattr(tabformer_adapter, "DATASET_METADATA_PATH", metadata_path)

    result = adapt_tabformer_archive(
        archive_path,
        output_path,
        chunksize=2,
        target_rows=4,
        seed=42,
    )
    metadata = read_json(metadata_path)
    expected_digest = hashlib.sha256(output_path.read_bytes()).hexdigest()

    assert len(result) == metadata["sample_rows"] == 4
    assert metadata["source_rows"] == 9
    assert metadata["eligible_positive_purchase_rows"] == 7
    assert metadata["excluded_non_positive_rows"] == 2
    assert metadata["sampling_uses_target"] is False
    assert metadata["dataset_sha256"] == expected_digest


def test_console_path_is_safe_for_legacy_windows_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class LegacyStdout:
        encoding = "cp1252"

    monkeypatch.setattr(tabformer_adapter.sys, "stdout", LegacyStdout())
    display = tabformer_adapter._console_safe_path(
        tmp_path / "문서" / "transactions.csv"
    )

    assert "문서" not in display
    assert "\\u" in display
