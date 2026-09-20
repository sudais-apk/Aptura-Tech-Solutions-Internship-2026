# Front End is Vibe coded

# Rain predictor: will it rain tomorrow?

A small web app around your Random Forest model from `model.ipynb`.
The person enters a city and today's weather; the app shows **Yes/No** and the **chance of rain in %**.

## Run it

```bash
pip install -r requirements.txt
python export_artifacts.py --data CleanData.csv   # trains the notebook's model and saves it (one time)
python app.py                                     # then open http://127.0.0.1:8000
```

Put `CleanData.csv` next to these files (or pass its path with `--data`).
Use the same Python environment for both commands, so the saved model loads cleanly.

## What the user types (9 numbers, a city and a date)

Lowest and highest temperature, humidity, air pressure, wind speed, cloud cover, and rain today / yesterday / in the last 7 days.
The **Fill in live weather** button fetches all of these from Open-Meteo (free, no key) using the city's coordinates from your data.

## What the app works out itself

Your model uses 23 features. The server derives the other 14 so nobody has to type them:

| Feature(s) | How |
|---|---|
| latitude, longitude, elevation, region | looked up from `CleanData.csv` by city |
| year, month, dayofweek, is_weekend, season | from the date (season and weekend rules are copied from your data) |
| tavg, temp_range | (tmin + tmax) / 2 and tmax − tmin |
| dew_point | Magnus formula from average temperature and humidity |
| rain_3day | exact when live weather is used; otherwise estimated from today, yesterday and the 7-day total |

## Check these assumptions against how you built CleanData.csv

1. `rain_3day` and `rain_7day` include today's rain. `export_artifacts.py` tests this and prints a warning if your data disagrees.
2. `pressure` is sea-level pressure in hPa and `wspd` is km/h. The app warns when a value falls outside the range seen in training.
3. The date entered is the day the readings were taken (the row's date), and the prediction is for the next day.
4. `dayofweek` counts Monday as 0, like pandas.

## Files

```
app.py                 Flask server and /api/predict
export_artifacts.py    trains the notebook's pipeline, saves model/ files
static/index.html      the whole front end (HTML, CSS, JS)
model/                 rain_model.joblib and metadata.json appear here
```
