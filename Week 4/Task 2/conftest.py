"""Shared helpers for all tests."""
import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg  # noqa: E402

TODAY = date(2026, 9, 20)   # fixed "today" so date tests never break

# A normal, valid request
VALID = {"city": "Islamabad", "date": "2024-07-15", "tmin": "26", "tmax": "34", "humidity": "80",
         "pressure": "1002", "cloud_cover": "90", "wspd": "12", "prcp": "5",
         "yesterday_prcp": "2", "rain_3day": "6", "rain_7day": "15"}


@pytest.fixture(scope="session")
def info():
    return json.loads(cfg.MODEL_INFO_FILE.read_text())


@pytest.fixture()
def valid():
    return dict(VALID)


@pytest.fixture()
def tmp_logs(tmp_path, monkeypatch):
    """Send the log files to a temporary folder so tests never touch the real logs."""
    monkeypatch.setattr(cfg, "PREDICTION_LOG", tmp_path / "predictions.csv")
    monkeypatch.setattr(cfg, "OUTCOME_LOG", tmp_path / "outcomes.csv")
    return tmp_path
