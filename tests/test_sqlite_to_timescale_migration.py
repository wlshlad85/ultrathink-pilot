import datetime as dt
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_persistence.migrations.sqlite_to_timescale import (
    compute_column_mapping,
    convert_value,
)


class TestComputeColumnMapping:
    def test_skips_missing_columns(self):
        usable, missing = compute_column_mapping(
            ["id", "name", "hyperparameters"],
            ["id", "name", "created_at"],
        )

        assert usable == [("id", "id"), ("name", "name")]
        assert ("hyperparameters", "hyperparameters") in missing

    def test_explicit_mapping_and_skip(self):
        usable, missing = compute_column_mapping(
            ["id", "model_id", "payload"],
            ["id", "checkpoint_id", "metadata"],
            {"model_id": "checkpoint_id", "payload": None},
        )

        assert usable == [("id", "id"), ("model_id", "checkpoint_id")]
        assert missing == [("payload", "<skipped>")]


class TestConvertValue:
    @pytest.mark.parametrize(
        "value",
        ["2024-01-02T03:04:05", "2024-01-02 03:04:05", "2024-01-02 03:04:05.123456"],
    )
    def test_converts_timestamps(self, value):
        converted = convert_value(value, "timestamp without time zone")
        assert isinstance(converted, dt.datetime)

    def test_passes_through_unparsed_timestamp(self):
        converted = convert_value("not-a-date", "timestamp without time zone")
        assert converted == "not-a-date"

    def test_converts_json_strings(self):
        converted = convert_value('{"a": 1}', "jsonb")
        assert converted == {"a": 1}

    @pytest.mark.parametrize("value, expected", [("1", 1), (2, 2)])
    def test_converts_integers(self, value, expected):
        assert convert_value(value, "integer") == expected

    @pytest.mark.parametrize("value, expected", [("1.5", 1.5), (2.75, 2.75)])
    def test_converts_reals(self, value, expected):
        assert convert_value(value, "double precision") == pytest.approx(expected)

    @pytest.mark.parametrize(
        "value, expected",
        [("true", True), ("False", False), (1, True), (0, False)],
    )
    def test_converts_booleans(self, value, expected):
        assert convert_value(value, "boolean") is expected
