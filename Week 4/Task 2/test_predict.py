"""Tests for src/predict.py  (the whole inference workflow)."""
import time

import pandas as pd
import pytest

from app.app import EXAMPLES
from src import config as cfg
from src.predict import load_model, predict_rain
from tests.conftest import TODAY


def run(raw, **kw):
    return predict_rain(raw, log=False, today=TODAY, **kw)


def test_result_has_all_expected_parts(valid):
    r = run(valid)
    for key in ["ok", "prediction", "answer", "rain_probability", "predicts_for", "sureness", "model_version", "warnings"]:
        assert key in r
    assert r["ok"] and r["predicts_for"] == "2024-07-16"


def test_probability_is_between_0_and_100(valid):
    assert 0 <= run(valid)["rain_probability"] <= 100


def test_yes_no_follows_the_threshold(valid):
    for tmax, hum, cloud, rain in [(34, 80, 90, 5), (40, 20, 0, 0), (28, 95, 100, 20), (33, 55, 40, 0)]:
        valid.update(tmax=str(tmax), humidity=str(hum), cloud_cover=str(cloud), prcp=str(rain))
        r = run(valid)
        if abs(r["rain_probability"] - cfg.THRESHOLD * 100) > 0.1:      # skip the tiny rounding zone
            assert r["prediction"] == int(r["rain_probability"] / 100 >= cfg.THRESHOLD)
        assert r["answer"] == ("Yes" if r["prediction"] else "No")


def test_same_input_gives_same_answer(valid):
    assert run(valid)["rain_probability"] == run(valid)["rain_probability"]


def test_bad_input_gives_errors_and_no_prediction(valid):
    valid["humidity"] = "500"
    r = run(valid)
    assert r["ok"] is False and "humidity" in r["errors"] and "prediction" not in r


@pytest.mark.parametrize("city", list(cfg.CITIES))
def test_every_city_works(valid, city):
    valid["city"] = city
    valid.update(tmin="15", tmax="27", wspd="4", pressure="1010")
    assert run(valid)["ok"]


@pytest.mark.parametrize("city", list(cfg.CITIES))
def test_wet_day_scores_higher_than_dry_day(city):
    """Common-sense check: cloudy, humid, rainy weather must give a higher chance than clear, dry weather."""
    base = {"city": city, "date": "2024-03-15", "tmin": "14", "tmax": "26", "pressure": "1008", "wspd": "4"}
    wet = run(dict(base, humidity="92", cloud_cover="95", prcp="10", yesterday_prcp="5", rain_3day="12", rain_7day="25"))
    dry = run(dict(base, humidity="25", cloud_cover="0", prcp="0", yesterday_prcp="0", rain_3day="0", rain_7day="0"))
    assert wet["rain_probability"] > dry["rain_probability"]


def test_the_two_example_buttons_give_the_expected_answers():
    assert run(EXAMPLES["rainy"])["answer"] == "Yes"
    assert run(EXAMPLES["dry"])["answer"] == "No"


def test_guessed_values_are_reported(valid):
    valid["rain_3day"] = ""
    r = run(valid)
    assert "rain_3day" in r["assumed"] and r["warnings"]


def test_prediction_is_fast_after_loading(valid):
    load_model()
    run(valid)                                     # warm up
    start = time.perf_counter()
    run(valid)
    assert (time.perf_counter() - start) < 0.5


def test_prediction_is_written_to_the_log(valid, tmp_logs):
    predict_rain(valid, log=True, today=TODAY)
    log = pd.read_csv(cfg.PREDICTION_LOG)
    assert len(log) == 1 and log.loc[0, "city"] == "Islamabad" and log.loc[0, "predicts_for"] == "2024-07-16"


def test_failed_input_is_not_logged(valid, tmp_logs):
    valid["humidity"] = "999"
    predict_rain(valid, log=True, today=TODAY)
    assert not cfg.PREDICTION_LOG.exists()


def test_log_problem_does_not_stop_the_answer(valid, tmp_logs, monkeypatch):
    bad_path = tmp_logs / "a_folder"
    bad_path.mkdir()
    monkeypatch.setattr(cfg, "PREDICTION_LOG", bad_path)      # a folder cannot be opened as a file
    r = predict_rain(valid, log=True, today=TODAY)
    assert r["ok"] and any("log" in w.lower() for w in r["warnings"])
