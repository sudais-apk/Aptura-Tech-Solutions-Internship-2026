"""Tests for src/monitoring.py  (logs, performance report, drift, alerts)."""
import pandas as pd

from src import config as cfg
from src.monitoring import (drift_report, log_outcome, log_prediction, make_alerts,
                            performance_report, read_logs, text_report)


def fake_predictions(n, prediction=1, rain_probability=70.0, **inputs):
    base = {"city": "Islamabad", "date": "2024-07-15", "predicts_for": "2024-07-16", "tmin": 26, "tmax": 34,
            "humidity": 80, "pressure": 1002, "cloud_cover": 90, "wspd": 12, "prcp": 5,
            "assumed_fields": "", "prediction": prediction, "rain_probability": rain_probability}
    base.update(inputs)
    return pd.DataFrame([base] * n)


def test_log_files_are_created_with_a_header(tmp_logs):
    result = {"model_version": "1.0.0", "predicts_for": "2024-07-16", "assumed": ["rain_3day"], "warnings": ["x"],
              "rain_probability": 61.5, "prediction": 1, "latency_ms": 9.9}
    log_prediction(result, {"city": "Lahore", "date": "2024-07-15", "tmin": 20}, path=cfg.PREDICTION_LOG)
    log_outcome("Lahore", "2024-07-16", 2.5, path=cfg.OUTCOME_LOG)
    preds, outs = read_logs()
    assert preds.loc[0, "assumed_fields"] == "rain_3day" and preds.loc[0, "warning_count"] == 1
    assert outs.loc[0, "rained"] == 1


def test_tiny_rain_counts_as_rain_like_the_training_data(tmp_logs):
    log_outcome("Lahore", "2024-07-16", 0.1)
    log_outcome("Lahore", "2024-07-17", 0.0)
    assert list(read_logs()[1]["rained"]) == [1, 0]


def test_performance_numbers_are_correct():
    preds = pd.DataFrame({
        "city": ["Lahore"] * 4, "predicts_for": ["d1", "d2", "d3", "d4"],
        "prediction": [1, 1, 0, 0], "rain_probability": [90, 80, 20, 10]})
    outs = pd.DataFrame({"city": ["Lahore"] * 4, "date": ["d1", "d2", "d3", "d4"], "rained": [1, 0, 1, 0]})
    r = performance_report(preds, outs)
    assert r["matched"] == 4 and r["accuracy"] == 0.5
    assert r["rain_recall"] == 0.5 and r["rain_precision"] == 0.5 and r["baseline_accuracy"] == 0.5


def test_no_results_yet_is_handled():
    assert performance_report(fake_predictions(3), pd.DataFrame(columns=["city", "date", "rained"])) == {"matched": 0}


def test_bad_performance_creates_alerts():
    perf = {"matched": 500, "accuracy": 0.60, "baseline_accuracy": 0.70, "rain_recall": 0.20}
    text = " ".join(make_alerts(perf, {"rows": 0}))
    assert "ACCURACY LOW" in text and "NOT BETTER THAN GUESSING" in text and "MISSED RAIN" in text


def test_good_performance_creates_no_alerts():
    perf = {"matched": 500, "accuracy": 0.82, "baseline_accuracy": 0.70, "rain_recall": 0.62}
    assert make_alerts(perf, {"rows": 0}) == []


def test_too_few_results_are_not_judged():
    perf = {"matched": 10, "accuracy": 0.1, "baseline_accuracy": 0.7, "rain_recall": 0.0}
    alerts = make_alerts(perf, {"rows": 0})
    assert len(alerts) == 1 and "Only 10 predictions" in alerts[0]


def test_normal_inputs_show_no_drift(info):
    means = {c: info["training_stats"]["Islamabad"]["7"][c][0] for c in cfg.DRIFT_COLS}
    d = drift_report(fake_predictions(60, **means), info)
    assert all(abs(v["mean_shift"]) < 0.05 for v in d["columns"].values())
    assert make_alerts({"matched": 0}, d) == []


def test_shifted_inputs_are_flagged(info):
    hot = fake_predictions(60, tmax=info["training_ranges"]["Islamabad"]["tmax"][1] + 3)
    text = " ".join(make_alerts({"matched": 0}, drift_report(hot, info)))
    assert "INPUT DRIFT" in text and "tmax" in text and "outside the training range" in text


def test_many_guessed_values_are_flagged(info):
    d = drift_report(fake_predictions(60, assumed_fields="rain_3day|rain_7day"), info)
    assert d["share_assumed"] == 1.0 and any("GUESSED" in a for a in make_alerts({"matched": 0}, d))


def test_text_report_can_be_built(info):
    text = text_report(fake_predictions(5), pd.DataFrame(columns=["city", "date", "rained"]), info)
    assert "MONITORING REPORT" in text and "ALERTS" in text
