# Rain Prediction: Model Deployment and Responsible Reporting

**Final Task 2.** This project takes the Random Forest model from `notebooks/model.ipynb`
("will it rain tomorrow?") and makes it ready for real use: a web page, input checks,
monitoring, tests, and honest documentation about what the model can and cannot do.

## Results in one look

| Question | Answer |
|---|---|
| Selected model | Random Forest (100 trees), same settings as the notebook |
| Notebook test score | **83.5% accuracy** (the Decision Tree got 73%) |
| A model that always says "no rain" gets | 77.9% on the same test, so the real gain is small |
| Rain days the model catches (recall) | 45% in the notebook test, 66% in the per-city test |
| Weakest city | Karachi: it catches only about 19% of rain days |
| Speed | about 10 ms per prediction (after the model is loaded) |
| Automatic tests | 116 passed, plus 9 of 9 browser checks passed |

Read `docs/MODEL_CARD.md` before you trust any answer from this model.

## Project structure

```
rain-prediction-deployment/
├── README.md                  <- you are here
├── Final_Report.docx          <- short final report (decisions, results, limits)
├── requirements.txt
├── data/
│   └── CleanData.csv          <- daily weather, 6 cities, 2000-2024 (31,779 rows)
├── notebooks/
│   └── model.ipynb            <- the original notebook (not changed)
├── src/                       <- the code that does the work
│   ├── config.py              <- all settings in one place
│   ├── train_model.py         <- tests the model and saves the final one
│   ├── validation.py          <- checks the user's inputs
│   ├── features.py            <- turns few inputs into the 23 model columns
│   ├── predict.py             <- the inference workflow (6 steps)
│   └── monitoring.py          <- logs, performance report, drift check, alerts
├── app/                       <- the web page (front end)
│   ├── app.py                 <- Flask app
│   ├── templates/index.html
│   └── static/style.css
├── models/
│   ├── rain_model.joblib      <- the saved model
│   └── model_info.json        <- city table, normal ranges, typical values
├── tests/                     <- 116 automatic tests (pytest)
├── scripts/
│   ├── monitoring_demo.py     <- simulation that shows the monitoring working
│   └── take_screenshots.py    <- browser checks + screenshots
├── reports/                   <- scores (csv/json), charts, monitoring demo output
├── docs/
│   ├── INFERENCE_AND_VALIDATION.md
│   ├── MODEL_CARD.md          <- limits, fairness and risk
│   ├── MONITORING_PLAN.md     <- how to track the model after deployment
│   ├── TEST_EVIDENCE.md
│   ├── test_results_full.txt  <- full pytest output
│   └── screenshots/           <- demo and test pictures
└── logs/                      <- the app writes predictions.csv here when it is used
```

## How to run it

You need Python 3.10 or newer.

```bash
# 1. Install the libraries
pip install -r requirements.txt

# 2. (Optional) train and test again. This rebuilds models/ and reports/
python -m src.train_model

# 3. Start the web page, then open http://127.0.0.1:5000
python app/app.py

# 4. Run the automatic tests
python -m pytest -v

# 5. Show the monitoring tools working (simulation)
python -m scripts.monitoring_demo

# 6. Print the monitoring report for the real log file
python -m src.monitoring

# 7. (Optional) redo the browser checks and screenshots
playwright install chromium
python -m scripts.take_screenshots
```

If you see a warning about scikit-learn versions, or the model file cannot be loaded,
run step 2. It builds a new model file with the version you have installed.

## The inference workflow

Every prediction, from the web page or from the API, goes through the same steps:

```
 User types values
        |
 1. RECEIVE    form or JSON
        |
 2. VALIDATE   missing? not a number? impossible? unusual?     (src/validation.py)
        |--- errors  -> stop, show clear messages, no answer
        |
 3. BUILD      fill in year/month/season, region/lat/long/elevation,
        |      tavg, temp_range, dew_point                     (src/features.py)
        |
 4. PREDICT    Random Forest gives the rain probability          (src/predict.py)
        |
 5. DECIDE     probability >= 50%  ->  "Yes, rain is likely tomorrow"
        |
 6. LOG        save the request and answer for monitoring        (src/monitoring.py)
        |
 Show: Yes/No, probability %, how sure the model is, warnings
```

More details and examples: `docs/INFERENCE_AND_VALIDATION.md`.

## API (for other programs)

```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" \
  -d '{"city":"Islamabad","date":"2024-07-15","tmin":25,"tmax":31,"humidity":88,
       "pressure":1000,"cloud_cover":95,"wspd":14,"prcp":12.4}'
```

Other endpoints: `GET /api/health` and `POST /api/outcome` (save what really happened).

## Screenshots

| | |
|---|---|
| ![Home page](docs/screenshots/01_home_page.png) | ![Answer: Yes](docs/screenshots/02_result_rain_yes.png) |
| Home page | Rainy example: "Yes, rain is likely" |
| ![Answer: No](docs/screenshots/03_result_rain_no.png) | ![Errors](docs/screenshots/05_impossible_values.png) |
| Dry example: "No" | Impossible values are refused |

More pictures are in `docs/screenshots/`.

## Things to know

* **Data files:** `CleanData.csv` has one row per day for each city. The city data stops in different years
  (Gilgit 2011, Peshawar 2013, Karachi 2015), so the model has no recent data for those cities.
* **Small bug in the original notebook (not changed):** cell 21 is meant to print the Decision Tree accuracy
  but it uses `y_pred_rf` (the Random Forest answers), so it prints 83.5% again. The comparison in cell 24
  is correct (Decision Tree 73%, Random Forest 83%).
* **The final model is trained on all rows.** The scores in the reports come from models that were trained
  on part of the data and tested on the rest. See `docs/MODEL_CARD.md`, section "How it was tested".
