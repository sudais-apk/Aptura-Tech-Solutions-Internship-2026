"""Tests for src/validation.py  (input validation)."""
import pytest

from src import config as cfg
from src.validation import validate_inputs
from tests.conftest import TODAY


def check(raw, info):
    return validate_inputs(raw, info, today=TODAY)


def test_good_input_passes(valid, info):
    r = check(valid, info)
    assert r["ok"] and r["errors"] == {}
    assert r["clean"]["tmin"] == 26.0 and r["assumed"] == []


@pytest.mark.parametrize("field", [f["name"] for f in cfg.INPUT_FIELDS if f["required"]] + ["city", "date"])
def test_required_field_missing(valid, info, field):
    valid.pop(field)
    r = check(valid, info)
    assert not r["ok"] and field in r["errors"]


def test_empty_form_gives_one_error_per_required_box(info):
    r = check({}, info)
    required = [f["name"] for f in cfg.INPUT_FIELDS if f["required"]]
    assert not r["ok"] and set(r["errors"]) == set(required + ["city", "date"])


def test_unknown_city(valid, info):
    valid["city"] = "Paris"
    assert "city" in check(valid, info)["errors"]


@pytest.mark.parametrize("bad", ["abc", "12abc", "nan", "inf", "-inf", "1e999", "--5"])
def test_not_a_number(valid, info, bad):
    valid["tmax"] = bad
    assert "tmax" in check(valid, info)["errors"]


@pytest.mark.parametrize("field,value", [
    ("humidity", "150"), ("humidity", "0"), ("cloud_cover", "-5"), ("cloud_cover", "101"),
    ("pressure", "500"), ("prcp", "-1"), ("wspd", "999"), ("tmin", "-80"), ("tmax", "90"),
])
def test_impossible_values_are_errors(valid, info, field, value):
    valid[field] = value
    assert field in check(valid, info)["errors"]


def test_lowest_temperature_above_highest(valid, info):
    valid["tmin"], valid["tmax"] = "35", "30"
    assert "tmin" in check(valid, info)["errors"]


@pytest.mark.parametrize("bad", ["15/07/2024", "2024-13-01", "2024-02-30", "yesterday", "", "20240715"])
def test_bad_date_format(valid, info, bad):
    valid["date"] = bad
    assert "date" in check(valid, info)["errors"]


def test_future_date_rejected(valid, info):
    valid["date"] = "2026-09-25"
    assert "date" in check(valid, info)["errors"]


def test_date_before_data_starts_rejected(valid, info):
    valid["date"] = "1999-12-31"
    assert "date" in check(valid, info)["errors"]


def test_today_is_accepted(valid, info):
    valid["date"] = TODAY.isoformat()
    assert check(valid, info)["ok"]


def test_rain_history_must_be_in_order(valid, info):
    valid.update({"yesterday_prcp": "10", "rain_3day": "4", "rain_7day": "20"})
    assert "rain_3day" in check(valid, info)["errors"]
    valid.update({"yesterday_prcp": "1", "rain_3day": "10", "rain_7day": "4"})
    assert "rain_7day" in check(valid, info)["errors"]


def test_empty_optional_boxes_use_typical_values(valid, info):
    for name in cfg.OPTIONAL_FIELDS:
        valid[name] = ""
    r = check(valid, info)
    assert r["ok"] and set(r["assumed"]) == set(cfg.OPTIONAL_FIELDS)
    assert any("typical values" in w for w in r["warnings"])


def test_guessed_values_never_break_the_order(valid, info):
    """User types only 3-day rain = 0. The guessed 'yesterday' must not be bigger than that."""
    valid["yesterday_prcp"], valid["rain_7day"] = "", ""
    valid["rain_3day"] = "0"
    r = check(valid, info)["clean"]
    assert r["yesterday_prcp"] <= r["rain_3day"] <= r["rain_7day"]


def test_user_typed_values_are_never_changed(valid, info):
    valid["yesterday_prcp"] = ""
    valid["rain_3day"] = "3"
    r = check(valid, info)["clean"]
    assert r["rain_3day"] == 3.0


def test_unusual_value_is_only_a_warning(valid, info):
    valid["city"], valid["tmax"] = "Karachi", "45"        # hotter than anything seen in Karachi
    r = check(valid, info)
    assert r["ok"] and any("outside what the model saw" in w for w in r["warnings"])


def test_comma_decimal_is_accepted(valid, info):
    valid["prcp"] = "12,5"
    assert check(valid, info)["clean"]["prcp"] == 12.5


def test_numbers_from_json_are_accepted(info):
    raw = {"city": "Lahore", "date": "2024-08-01", "tmin": 27, "tmax": 36, "humidity": 70,
           "pressure": 1000, "cloud_cover": 50, "wspd": 10, "prcp": 0}
    assert check(raw, info)["ok"]


def test_true_and_false_are_not_numbers(valid, info):
    valid["humidity"] = True
    assert "humidity" in check(valid, info)["errors"]
