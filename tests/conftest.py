"""Reusable safe transaction fixtures."""

from __future__ import annotations

import json

import pytest

from config import SAMPLE_TRANSACTIONS_PATH


@pytest.fixture(scope="session")
def sample_transactions() -> dict:
    return json.loads(SAMPLE_TRANSACTIONS_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def normal_transaction(sample_transactions: dict) -> dict:
    return dict(sample_transactions["normal"])


@pytest.fixture()
def high_transaction(sample_transactions: dict) -> dict:
    return dict(sample_transactions["high"])
