"""Checks on the saved model and the saved scores."""
import json

import pytest
import sklearn

from src import config as cfg
from src.predict import load_model

NOTEBOOK_ACCURACY = 0.8349590937696665      # printed in model.ipynb for the Random Forest


def test_files_exist():
    assert cfg.MODEL_FILE.exists() and cfg.MODEL_INFO_FILE.exists() and (cfg.REPORTS_DIR / "metrics.json").exists()


def test_model_is_a_random_forest_pipeline():
    model, info = load_model()
    assert list(model.named_steps) == ["preprocessing", "model"]
    assert type(model.named_steps["model"]).__name__ == "RandomForestClassifier"
    assert list(model.classes_) == [0, 1]


def test_info_matches_config(info):
    assert info["model_version"] == cfg.MODEL_VERSION
    assert set(info["cities"]) == set(cfg.CITIES)
    assert info["threshold"] == cfg.THRESHOLD


def test_we_reproduce_the_notebook_score():
    metrics = json.loads((cfg.REPORTS_DIR / "metrics.json").read_text())["notebook_test"]
    assert metrics["accuracy"] == pytest.approx(NOTEBOOK_ACCURACY, abs=0.0005)


def test_model_beats_the_always_no_rain_guess():
    metrics = json.loads((cfg.REPORTS_DIR / "metrics.json").read_text())
    for name in ["notebook_test", "per_city_test_overall"]:
        assert metrics[name]["accuracy"] > metrics[name]["baseline_accuracy"], name


def test_scikit_learn_version_is_the_one_used_for_training(info):
    assert info["sklearn_version"].split(".")[:2] == sklearn.__version__.split(".")[:2], \
        "Model was saved with another scikit-learn version. Run: python -m src.train_model"
