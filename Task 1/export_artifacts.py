"""
Train the Random Forest exactly as in model.ipynb and save what the web app needs:

    model/rain_model.joblib   the fitted pipeline (preprocessing + Random Forest)
    model/metadata.json       cities, season/weekend lookups, value ranges, test metrics

Usage:
    python export_artifacts.py --data CleanData.csv

Run it with the same Python environment you will use for app.py, so the
scikit-learn version that saves the model is the one that loads it.
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLS = [
    "year", "month", "dayofweek", "is_weekend", "latitude", "longitude", "elevation",
    "tmin", "tmax", "tavg", "prcp", "wspd", "humidity", "pressure", "dew_point",
    "cloud_cover", "temp_range", "yesterday_prcp", "rain_3day", "rain_7day",
]
CATEGORICAL_COLS = ["season", "city", "region"]
RANGE_COLS = ["tmin", "tmax", "humidity", "pressure", "wspd", "cloud_cover",
              "prcp", "yesterday_prcp", "rain_7day"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="CleanData.csv", help="path to CleanData.csv")
    ap.add_argument("--out", default="model", help="output folder")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = df.drop(columns=["will_rain", "tomorrow_prcp"])
    y = df["will_rain"]

    # Same pipeline and same time-ordered 80/20 split as the notebook.
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    split = int(len(df) * 0.8)
    pipeline.fit(X.iloc[:split], y.iloc[:split])
    pred = pipeline.predict(X.iloc[split:])
    y_test = y.iloc[split:]
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, average="weighted"),
        "recall": recall_score(y_test, pred, average="weighted"),
        "f1": f1_score(y_test, pred, average="weighted"),
    }
    print("Test metrics:", {k: round(v * 100, 2) for k, v in metrics.items()})

    # Lookups the app uses to fill in features the user should not have to type.
    cities = (
        df.groupby("city")
        .agg(region=("region", lambda s: s.mode().iloc[0]),
             latitude=("latitude", "mean"),
             longitude=("longitude", "mean"),
             elevation=("elevation", "mean"))
        .reset_index()
        .round({"latitude": 4, "longitude": 4, "elevation": 1})
        .sort_values("city")
    )
    season_by_month = {str(int(m)): str(g.mode().iloc[0]) for m, g in df.groupby("month")["season"]}
    weekend_by_dow = {str(int(d)): int(g.mode().iloc[0]) for d, g in df.groupby("dayofweek")["is_weekend"]}
    ranges = {c: [float(df[c].min()), float(df[c].max())] for c in RANGE_COLS}

    # The app assumes rain_3day / rain_7day are rolling sums that include today's rain.
    includes_today = float((df["rain_3day"] + 1e-6 >= df["prcp"] + df["yesterday_prcp"]).mean())
    if includes_today > 0.98:
        print("Check passed: rain_3day includes today's and yesterday's rain, as the app assumes.")
    else:
        print(f"WARNING: rain_3day >= prcp + yesterday_prcp in only {includes_today:.0%} of rows. "
              "The app assumes rolling sums that include today; check how you built these columns.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out / "rain_model.joblib", compress=3)
    metadata = {
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "feature_columns": list(X.columns),
        "cities": cities.to_dict(orient="records"),
        "season_by_month": season_by_month,
        "weekend_by_dow": weekend_by_dow,
        "ranges": ranges,
        "metrics": metrics,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model and metadata to {out.resolve()}")


if __name__ == "__main__":
    main()
