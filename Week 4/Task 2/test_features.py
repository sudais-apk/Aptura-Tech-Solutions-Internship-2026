"""Tests for src/features.py  (turning few inputs into the 23 model columns)."""
import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.features import build_model_row, dew_point, derive_temperature_columns, season_of
from src.validation import validate_inputs
from tests.conftest import TODAY, VALID


def clean_row(**changes):
    raw = dict(VALID, **changes)
    return validate_inputs(raw, None, today=TODAY)["clean"]


def test_row_has_exactly_the_training_columns():
    row = build_model_row(clean_row())
    assert list(row.columns) == cfg.NUMERIC_COLS + cfg.CATEGORICAL_COLS
    assert row.shape == (1, 23)


def test_temperature_helpers():
    tavg, rng, dp = derive_temperature_columns(20, 30, 50)
    assert tavg == 25 and rng == 10 and 10 < dp < 20


@pytest.mark.parametrize("month,season", [(12, "Winter"), (1, "Winter"), (2, "Winter"), (3, "Spring"), (5, "Spring"),
                                          (6, "Summer"), (8, "Summer"), (9, "Autumn"), (11, "Autumn")])
def test_season(month, season):
    assert season_of(month) == season


def test_date_columns_match_the_dataset():
    """2000-01-01 was a Saturday. The first row of CleanData.csv says dayofweek=5, is_weekend=1."""
    first = pd.read_csv(cfg.DATA_FILE, nrows=1).iloc[0]
    row = build_model_row(clean_row(date="2000-01-01")).iloc[0]
    assert (row["dayofweek"], row["is_weekend"], row["month"], row["year"]) == \
           (first["dayofweek"], first["is_weekend"], first["month"], first["year"])
    assert row["season"] == first["season"]


def test_weekday_is_not_weekend():
    assert build_model_row(clean_row(date="2024-07-15")).iloc[0]["is_weekend"] == 0   # a Monday


def test_city_table_matches_the_csv():
    df = pd.read_csv(cfg.DATA_FILE)
    table = df.groupby("city")[["region", "latitude", "longitude", "elevation"]].first()
    for city, values in cfg.CITIES.items():
        for key, expected in values.items():
            assert table.loc[city, key] == expected, (city, key)


def test_worked_out_columns_are_close_to_the_real_ones():
    """tavg / temp_range / dew_point that we calculate should be close to the values in the data."""
    df = pd.read_csv(cfg.DATA_FILE)
    tavg, rng, dp = derive_temperature_columns(df["tmin"], df["tmax"], df["humidity"])
    assert np.abs(rng - df["temp_range"]).max() < 0.01
    assert np.abs(tavg - df["tavg"]).mean() < 1.0
    assert np.abs(dp - df["dew_point"]).mean() < 1.5


def test_dew_point_never_above_temperature():
    assert dew_point(30, 100) == pytest.approx(30, abs=0.01)
    assert dew_point(30, 40) < 30
