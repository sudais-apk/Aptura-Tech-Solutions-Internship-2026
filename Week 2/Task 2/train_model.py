"""
Car Price Prediction — Baseline Supervised Regression Model
==============================================================
Task type   : Regression (target = Price, a continuous variable)
Dataset     : car_price_prediction_.csv (2500 rows, 10 columns)

Pipeline:
1. Load & inspect data
2. Train/test split (done BEFORE any preprocessing to avoid leakage)
3. Preprocess features (numeric scaling + categorical one-hot encoding)
   -- fit only on training data, applied to test data
4. Train two baseline models: Linear Regression and Random Forest
5. Evaluate with MAE, RMSE, R^2 + cross-validation
6. Save plots, metrics, and trained model artifacts
"""

import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

DATA_PATH = "data/car_price_prediction_.csv"
PLOTS_DIR = "plots"
OUT_DIR = "outputs"

# ----------------------------------------------------------------------------
# 1. Load & inspect
# ----------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values per column:\n", df.isnull().sum())

# Car ID is a pure identifier — no predictive value, drop it.
df = df.drop(columns=["Car ID"])

TARGET = "Price"
numeric_features = ["Year", "Engine Size", "Mileage"]
categorical_features = ["Brand", "Fuel Type", "Transmission", "Condition", "Model"]

# ----------------------------------------------------------------------------
# EDA plots (saved, not shown)
# ----------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.histplot(df[TARGET], kde=True, bins=40, color="#4C72B0")
plt.title("Distribution of Car Price")
plt.xlabel("Price")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/price_distribution.png", dpi=140)
plt.close()

plt.figure(figsize=(8, 6))
corr = df[numeric_features + [TARGET]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (numeric features)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/correlation_heatmap.png", dpi=140)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x="Brand", y=TARGET, ax=axes[0])
axes[0].set_title("Price by Brand")
axes[0].tick_params(axis="x", rotation=45)
sns.boxplot(data=df, x="Condition", y=TARGET, ax=axes[1])
axes[1].set_title("Price by Condition")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/price_by_category.png", dpi=140)
plt.close()

# ----------------------------------------------------------------------------
# 2. Train/test split BEFORE preprocessing (avoid data leakage)
# ----------------------------------------------------------------------------
X = df[numeric_features + categorical_features]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ----------------------------------------------------------------------------
# 3. Preprocessing pipeline (fit on train only, via sklearn Pipeline)
# ----------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ----------------------------------------------------------------------------
# 4. Baseline models
# ----------------------------------------------------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=200, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    ),
}

results = {}
fitted_pipelines = {}

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    # 5-fold cross-validation on the TRAINING set only
    cv_scores = cross_val_score(
        pipe, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error"
    )
    cv_rmse_mean = -cv_scores.mean()
    cv_rmse_std = cv_scores.std()

    # Fit on full training set, evaluate on held-out test set
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results[name] = {
        "cv_rmse_mean": cv_rmse_mean,
        "cv_rmse_std": cv_rmse_std,
        "test_MAE": mae,
        "test_RMSE": rmse,
        "test_R2": r2,
    }
    fitted_pipelines[name] = pipe

    print(f"\n--- {name} ---")
    print(f"CV RMSE: {cv_rmse_mean:.2f} (+/- {cv_rmse_std:.2f})")
    print(f"Test MAE : {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R^2 : {r2:.4f}")

# ----------------------------------------------------------------------------
# Also compute a naive baseline (predict the mean price) for comparison
# ----------------------------------------------------------------------------
naive_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
naive_mae = mean_absolute_error(y_test, naive_pred)
naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
naive_r2 = r2_score(y_test, naive_pred)
results["NaiveMeanBaseline"] = {
    "cv_rmse_mean": None,
    "cv_rmse_std": None,
    "test_MAE": naive_mae,
    "test_RMSE": naive_rmse,
    "test_R2": naive_r2,
}
print(f"\n--- Naive Mean Baseline ---")
print(f"Test MAE : {naive_mae:.2f}")
print(f"Test RMSE: {naive_rmse:.2f}")
print(f"Test R^2 : {naive_r2:.4f}")

# ----------------------------------------------------------------------------
# Save metrics as JSON
# ----------------------------------------------------------------------------
with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

# ----------------------------------------------------------------------------
# Pick best model by test RMSE (excluding naive baseline) and save it
# ----------------------------------------------------------------------------
best_name = min(
    (n for n in results if n != "NaiveMeanBaseline"),
    key=lambda n: results[n]["test_RMSE"],
)
best_pipe = fitted_pipelines[best_name]
joblib.dump(best_pipe, f"{OUT_DIR}/best_model_{best_name}.joblib")
print(f"\nBest model: {best_name} -> saved to {OUT_DIR}/best_model_{best_name}.joblib")

# ----------------------------------------------------------------------------
# Predicted vs Actual plot for the best model
# ----------------------------------------------------------------------------
best_preds = best_pipe.predict(X_test)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_preds, alpha=0.4, s=15, color="#55A868")
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Predicted vs Actual — {best_name}")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/predicted_vs_actual.png", dpi=140)
plt.close()

# Residuals plot
residuals = y_test.values - best_preds
plt.figure(figsize=(7, 5))
sns.histplot(residuals, kde=True, bins=40, color="#C44E52")
plt.title(f"Residuals Distribution — {best_name}")
plt.xlabel("Residual (Actual - Predicted)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/residuals_distribution.png", dpi=140)
plt.close()

# Feature importance (Random Forest only)
if "RandomForest" in fitted_pipelines:
    rf_pipe = fitted_pipelines["RandomForest"]
    ohe = rf_pipe.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_names
    importances = rf_pipe.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame(
        {"feature": all_feature_names, "importance": importances}
    ).sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", color="#4C72B0")
    plt.title("Top 15 Feature Importances — Random Forest")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/feature_importance.png", dpi=140)
    plt.close()

print("\nAll plots saved to", PLOTS_DIR)
print("Metrics saved to", f"{OUT_DIR}/metrics.json")
print("Done.")
