"""Tests for the web app (app/app.py) using Flask's built-in test client."""
import pandas as pd
import pytest

from app.app import EXAMPLES, create_app
from src import config as cfg


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_health_check(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_home_page_shows_the_form(client):
    html = client.get("/").get_data(as_text=True)
    assert "Will it rain tomorrow?" in html and "Predict rain" in html
    for city in cfg.CITIES:
        assert f'value="{city}"' in html


def test_form_with_good_values_shows_an_answer(client, tmp_logs):
    r = client.post("/predict", data=EXAMPLES["rainy"])
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and "chance of rain" in html and "Yes, rain is likely tomorrow" in html


def test_form_with_bad_values_shows_errors_and_no_answer(client, tmp_logs):
    r = client.post("/predict", data=dict(EXAMPLES["rainy"], humidity="150", tmin="40", tmax="30"))
    html = r.get_data(as_text=True)
    assert r.status_code == 400
    assert "Please fix 2 problems" in html and "Humidity must be between 1 and 100" in html
    assert "chance of rain" not in html
    assert 'value="150"' in html            # what the user typed is kept, so they can fix it


def test_empty_form_shows_all_required_errors(client):
    html = client.post("/predict", data={}).get_data(as_text=True)
    assert "Please choose a city." in html and "Rain today is required." in html


def test_page_escapes_dangerous_text(client):
    html = client.post("/predict", data=dict(EXAMPLES["dry"], tmin="<script>alert(1)</script>")).get_data(as_text=True)
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html


def test_api_good_request(client, tmp_logs):
    r = client.post("/api/predict", json={k: v for k, v in EXAMPLES["dry"].items()})
    body = r.get_json()
    assert r.status_code == 200 and body["ok"] and body["answer"] == "No" and 0 <= body["rain_probability"] <= 100


def test_api_bad_request(client):
    r = client.post("/api/predict", json={"city": "Karachi"})
    assert r.status_code == 400 and not r.get_json()["ok"] and "date" in r.get_json()["errors"]


@pytest.mark.parametrize("payload", ["not json", "[1,2,3]", ""])
def test_api_wrong_body(client, payload):
    r = client.post("/api/predict", data=payload, content_type="application/json")
    assert r.status_code == 400


def test_api_saves_the_prediction_in_the_log(client, tmp_logs):
    client.post("/api/predict", json=EXAMPLES["rainy"])
    assert len(pd.read_csv(cfg.PREDICTION_LOG)) == 1


def test_outcome_endpoint(client, tmp_logs):
    ok = client.post("/api/outcome", json={"city": "Karachi", "date": "2024-01-21", "rain_mm": 0.4})
    assert ok.status_code == 200
    row = pd.read_csv(cfg.OUTCOME_LOG).iloc[0]
    assert row["rained"] == 1
    for bad in [{"city": "Nowhere", "date": "2024-01-21", "rain_mm": 1}, {"city": "Karachi", "date": "bad", "rain_mm": 1},
                {"city": "Karachi", "date": "2024-01-21", "rain_mm": -3}, {}]:
        assert client.post("/api/outcome", json=bad).status_code == 400


def test_huge_request_is_refused(client):
    assert client.post("/api/predict", data="x" * 50_000, content_type="application/json").status_code == 413
