"""
Rain prediction web app.

Serves the front end (static/index.html) and two JSON endpoints:
    GET  /api/info      cities and model accuracy, used to build the form
    POST /api/predict   takes the few inputs a person can type, works out the
                        remaining model features, and returns the prediction

Run:  python app.py      then open http://127.0.0.1:8000
"""
import datetime as dt
import json
import math
import os
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from flask import Flask, jsonify, request

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model" / "rain_model.joblib"
META_PATH = BASE / "model" / "metadata.json"

app = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="")

MODEL = None
META = None
CITIES = {}
POS_INDEX = 1
LOAD_ERROR = None

# key: (min, max, friendly name). These are the numbers a person types in.
INPUTS = {
    "tmin": (-60, 60, "Lowest temperature"),
    "tmax": (-60, 60, "Highest temperature"),
    "humidity": (0, 100, "Humidity"),
    "pressure": (850, 1100, "Air pressure"),
    "wspd": (0, 300, "Wind speed"),
    "cloud_cover": (0, 100, "Cloud cover"),
    "prcp": (0, 1000, "Rain today"),
    "yesterday_prcp": (0, 1000, "Rain yesterday"),
    "rain_7day": (0, 3000, "Rain in the last 7 days"),
}


def load_artifacts():
    """Load the saved model and metadata once, at start-up."""
    global MODEL, META, CITIES, POS_INDEX, LOAD_ERROR
    try:
        MODEL = joblib.load(MODEL_PATH)
        META = json.loads(META_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        LOAD_ERROR = ("The trained model was not found. Run "
                      "'python export_artifacts.py --data CleanData.csv' first.")
        return
    CITIES = {c["city"]: c for c in META["cities"]}

    classes = list(MODEL.classes_)
    POS_INDEX = len(classes) - 1
    for want in (1, True, "1", "Yes", "yes", "Rain", "rain"):
        hit = [i for i, c in enumerate(classes) if c == want]
        if hit:
            POS_INDEX = hit[0]
            break

    if META.get("sklearn_version") != sklearn.__version__:
        print(f"Note: model saved with scikit-learn {META.get('sklearn_version')}, "
              f"running {sklearn.__version__}. Re-run export_artifacts.py if you see errors.")


def dew_point(temp_c, rh):
    """Magnus formula. Dew point in °C from temperature (°C) and relative humidity (%)."""
    a, b = 17.62, 243.12
    gamma = math.log(max(rh, 1.0) / 100.0) + a * temp_c / (b + temp_c)
    return b * gamma / (a - gamma)


def parse_inputs(payload):
    """Validate the typed values. Returns (values, field_errors)."""
    errors, values = {}, {}
    for key, (lo, hi, _name) in INPUTS.items():
        try:
            val = float(payload.get(key))
            if not math.isfinite(val):
                raise ValueError
        except (TypeError, ValueError):
            errors[key] = "Enter a number."
            continue
        if not lo <= val <= hi:
            errors[key] = f"Enter a value between {lo:g} and {hi:g}."
            continue
        values[key] = val

    if "tmin" in values and "tmax" in values and values["tmax"] < values["tmin"]:
        errors["tmax"] = "Highest temperature must be at least the lowest."
    if {"prcp", "yesterday_prcp", "rain_7day"} <= values.keys():
        floor = values["prcp"] + values["yesterday_prcp"]
        if values["rain_7day"] + 0.05 < floor:
            errors["rain_7day"] = f"Can't be less than today plus yesterday ({floor:.1f} mm)."
    return values, errors


def build_features(city_row, date, v, rain_3day):
    """Turn the typed inputs into the full feature row the model was trained on."""
    tavg = (v["tmin"] + v["tmax"]) / 2
    if rain_3day is None:
        # Estimate the day before yesterday as the average of the remaining days.
        rest = max(0.0, v["rain_7day"] - v["prcp"] - v["yesterday_prcp"])
        rain_3day = v["prcp"] + v["yesterday_prcp"] + rest / 5
    rain_3day = min(max(rain_3day, v["prcp"] + v["yesterday_prcp"]), v["rain_7day"])

    dow = date.weekday()  # Monday = 0, same as pandas .dt.dayofweek
    row = {
        "year": date.year,
        "month": date.month,
        "dayofweek": dow,
        "is_weekend": META["weekend_by_dow"].get(str(dow), int(dow >= 5)),
        "latitude": city_row["latitude"],
        "longitude": city_row["longitude"],
        "elevation": city_row["elevation"],
        "tmin": v["tmin"],
        "tmax": v["tmax"],
        "tavg": tavg,
        "prcp": v["prcp"],
        "wspd": v["wspd"],
        "humidity": v["humidity"],
        "pressure": v["pressure"],
        "dew_point": dew_point(tavg, v["humidity"]),
        "cloud_cover": v["cloud_cover"],
        "temp_range": v["tmax"] - v["tmin"],
        "yesterday_prcp": v["yesterday_prcp"],
        "rain_3day": rain_3day,
        "rain_7day": v["rain_7day"],
        "season": META["season_by_month"][str(date.month)],
        "city": city_row["city"],
        "region": city_row["region"],
    }
    return row


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/info")
def info():
    if MODEL is None:
        return jsonify(error=LOAD_ERROR), 503
    return jsonify(cities=META["cities"], metrics=META["metrics"])


@app.post("/api/predict")
def predict():
    if MODEL is None:
        return jsonify(error=LOAD_ERROR), 503
    payload = request.get_json(silent=True) or {}

    values, errors = parse_inputs(payload)

    city_row = CITIES.get(payload.get("city"))
    if city_row is None:
        errors["city"] = "Choose a city from the list."
    try:
        date = dt.date.fromisoformat(str(payload.get("date")))
    except ValueError:
        errors["date"] = "Enter a valid date."
    if errors:
        return jsonify(error="Please fix the highlighted fields.", fields=errors), 400

    try:
        rain_3day = float(payload["rain_3day"]) if payload.get("rain_3day") is not None else None
    except (TypeError, ValueError):
        rain_3day = None

    row = build_features(city_row, date, values, rain_3day)
    X = pd.DataFrame([row])[META["feature_columns"]]

    label = MODEL.predict(X)[0]
    proba = float(MODEL.predict_proba(X)[0][POS_INDEX])
    will_rain = bool(label == MODEL.classes_[POS_INDEX])

    warnings = []
    for key, (lo, hi) in META.get("ranges", {}).items():
        if key in values and not lo <= values[key] <= hi:
            name = INPUTS[key][2]
            warnings.append(f"{name} ({values[key]:g}) is outside the range the model was trained on "
                            f"({lo:.1f} to {hi:.1f}), so treat this result with extra caution.")

    return jsonify(
        will_rain=will_rain,
        probability=round(proba * 100, 1),
        warnings=warnings,
        derived={
            "Average temperature": f"{row['tavg']:.1f} °C",
            "Dew point": f"{row['dew_point']:.1f} °C",
            "Rain in the last 3 days": f"{row['rain_3day']:.1f} mm",
            "Season": row["season"],
            "Region": row["region"],
        },
    )


load_artifacts()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), debug=False)
