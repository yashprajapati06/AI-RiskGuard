"""Tests for JSON robustness helpers."""

import pytest

from src.utils import parse_json_list, read_json, validate_model_metadata


def test_read_json_rejects_non_object(tmp_path) -> None:
    path = tmp_path / "invalid_shape.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="Expected a JSON object"):
        read_json(path)


def test_malformed_stored_json_list_is_safe() -> None:
    assert parse_json_list("not-json") == []
    assert parse_json_list('{"wrong": "shape"}') == []
    assert parse_json_list('["NEW_DEVICE"]') == ["NEW_DEVICE"]


def test_model_metadata_rejects_incomplete_metrics() -> None:
    with pytest.raises(ValueError, match="missing"):
        validate_model_metadata({})
