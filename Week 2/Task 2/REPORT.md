# Car Price Prediction — Baseline Model Report

## 1. Problem Framing

**Task type:** Regression. The target column, `Price`, is a continuous numeric value, so this is treated as a supervised regression problem rather than classification.

**Dataset:** `car_price_prediction_.csv` — 2,500 rows, 10 columns, no missing values.

| Column | Type | Role |
|---|---|---|
| Car ID | int | Identifier — dropped (no predictive value) |
| Brand | categorical | Feature |
| Year | numeric | Feature |
| Engine Size | numeric | Feature |
| Fuel Type | categorical | Feature |
| Transmission | categorical | Feature |
| Mileage | numeric | Feature |
| Condition | categorical | Feature |
| Model | categorical | Feature |
| **Price** | numeric | **Target** |

## 2. Approach

1. **Split first:** the data was split into 80% train / 20% test (2,000 / 500 rows) *before* any preprocessing was fit, to prevent test-set information leaking into the model.
2. **Preprocessing** (fit only on the training split, via an `sklearn` `Pipeline` + `ColumnTransformer`):
   - Numeric features (`Year`, `Engine Size`, `Mileage`) → standardized (zero mean, unit variance).
   - Categorical features (`Brand`, `Fuel Type`, `Transmission`, `Condition`, `Model`) → one-hot encoded.
3. **Baseline models trained:**
   - Linear Regression (simple, interpretable baseline)
   - Random Forest Regressor, 200 trees (captures non-linear relationships)
4. **Evaluation:**
   - 5-fold cross-validation RMSE on the training set (checks stability of the estimate)
   - Held-out test set: MAE, RMSE, R²
   - A **naive baseline** (always predict the training mean price) was included for comparison — a model is only useful if it beats this.

## 3. Results

| Model | CV RMSE (train) | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Linear Regression | 27,528 ± 457 | 23,877 | 27,794 | **-0.020** |
| Random Forest | 28,221 ± 596 | 24,535 | 28,439 | **-0.068** |
| Naive mean baseline | — | 23,788 | 27,538 | -0.001 |

*(See `plots/` for the price distribution, correlation heatmap, price-by-category boxplots, predicted-vs-actual scatter, residuals histogram, and Random Forest feature importances.)*

**Headline finding:** neither trained model outperforms the naive "always predict the average price" baseline. R² is at or below zero for every model, meaning the models explain essentially none of the variance in price — the predicted-vs-actual plot shows predictions clustering tightly around the mean regardless of the true price.

## 4. Why This Happened — Root Cause

A quick correlation check explains the result:

| Feature | Correlation with Price |
|---|---|
| Year | -0.037 |
| Engine Size | -0.004 |
| Mileage | -0.009 |

Group means of `Price` by `Brand` and by `Condition` are also nearly identical across categories (all within ~2,500 of each other, against a price range of 5,000–100,000). In short, **the `Price` column in this dataset does not appear to be meaningfully derived from any of the other columns** — it behaves as if generated independently (e.g. randomly) of the features provided.

## 5. Limitations

- **No learnable signal detected.** This is the primary limitation: with near-zero correlations across the board, no baseline (or advanced) supervised model can be expected to predict `Price` accurately from these features.
- **Small feature set.** Only 8 raw features are available (after dropping the ID); real-world car pricing typically also depends on region, seller type, accident history, trim level, and market conditions, none of which are present here.
- **No hyperparameter tuning was performed** — this is intentionally a baseline; tuning would not fix a lack of underlying signal.
- **Random Forest performed worse than Linear Regression on the held-out test set**, consistent with overfitting noise in the training data rather than learning real structure.

## 6. Key Lessons & Next Steps

1. **Check for signal before investing in modeling.** A five-minute correlation/group-mean check at the start of any project can save significant time — it would have flagged this dataset's lack of signal immediately.
2. **An honest negative result is still a valid deliverable.** The correct conclusion here is not "the model needs to be more complex," but "this dataset does not support price prediction as currently constructed."
3. **For a real next iteration:**
   - Source or request a dataset where price is genuinely derived from car attributes (e.g., real marketplace listings).
   - If this dataset must be used, consider engineered features (car age, mileage-per-year, brand×model interactions) as a next check — though initial evidence suggests they are unlikely to help.
   - If a genuine relationship is later confirmed, escalate to gradient boosting (XGBoost/LightGBM) with proper hyperparameter search and hold-out validation.
4. **Testing evidence:** used 5-fold cross-validation on the training set to check that CV RMSE was consistent with the final held-out test RMSE (no large gap), confirming the (lack of) performance is stable and not a fluke of one particular split.

## 7. Deliverables in This Submission

- `train_model.py` — standalone, reproducible script running the full pipeline.
- `car_price_prediction_notebook.ipynb` — the same pipeline as an executed Jupyter notebook with inline plots and commentary.
- `outputs/metrics.json` — all evaluation metrics.
- `outputs/best_model_LinearRegression.joblib` — the saved, trained best-performing pipeline (preprocessing + model), ready to load with `joblib.load(...)`.
- `plots/` — price distribution, correlation heatmap, price-by-category boxplots, predicted-vs-actual, residuals distribution, and feature importance charts.
- `REPORT.md` — this document.
