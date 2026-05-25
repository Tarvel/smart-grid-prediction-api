# Smart Grid Hybrid AI Energy Prediction Service

This project runs a local FastAPI service and browser dashboard for testing the
latest model artifacts in `New folder`.

The current model set predicts energy in **Wh**. API and MQTT output fields now
use explicit `*_wh` names by default, so testers do not see misleading kW/kWh
labels.

## What Is In This Folder

| Path | Purpose |
|---|---|
| `new_prediction_api.py` | Main FastAPI + MQTT service. Loads the new model artifacts and serves the dashboard. |
| `test_prediction_api.py` | Compatibility launcher. Run this if older instructions mention it. |
| `test_dashboard.html` | Browser dashboard served at `/`. |
| `diagnose.py` | Terminal diagnostic script that checks the model over several CSV ranges. |
| `requirements.txt` | Python dependencies. |
| `New folder/datarig40.csv` | Default CSV for `/predict_next` and the dashboard Auto Next button. |
| `New folder/RIGDATA_20k.csv` | Optional CSV selectable with `CSV_FILE=RIGDATA_20k.csv`. |
| `New folder/RIGDATA_40k.csv` | Optional CSV selectable with `CSV_FILE=RIGDATA_40k.csv`. |
| `New folder/lgb_model.joblib` | LightGBM model. |
| `New folder/scaler_lgb.joblib` | LightGBM feature scaler. |
| `New folder/scaler_gru.joblib` | TE-GRU feature scaler. |
| `New folder/tegru_model.tflite` | TE-GRU model used by the API. |

## Active CSV For Auto Next

The dashboard's **Auto Next** button calls:

```text
GET /predict_next
```

That endpoint reads from the active `CSV_PATH` loaded by `new_prediction_api.py`.
By default, the active file is:

```text
New folder/datarig40.csv
```

You can confirm the active CSV at any time:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/metadata
```

Or in a browser:

```text
http://127.0.0.1:5000/metadata
```

To switch the Auto Next CSV before starting the server:

```powershell
$env:CSV_FILE = "RIGDATA_20k.csv"
python test_prediction_api.py
```

Valid shortcut values are:

```text
datarig40.csv
RIGDATA_20k.csv
RIGDATA_40k.csv
```

You can also point to a full custom path:

```powershell
$env:CSV_PATH = "C:\Users\HP\Downloads\testing2\New folder\RIGDATA_40k.csv"
python test_prediction_api.py
```

## Model Input Contract

The new notebook trained the models with this contract:

| Item | Value |
|---|---|
| Energy unit | `Wh` |
| TE-GRU sequence length | `48` rows |
| Feature history needed | `24` rows for lag/rolling features |
| Runtime window size | `72` previous/current rows |
| LightGBM features | 12 scaled non-lag features |
| TE-GRU features | 24 scaled lag + rolling + sensor/time features |
| TE-GRU inputs | numeric sequence plus separate hour input |

Supported CSV input schemas:

| CSV style | Expected columns |
|---|---|
| `datarig40.csv` | `timestamp`, `time_of_day`, `day_of_week`, `temperature`, `humidity`, `lux`, `occupancy`, `energy` |
| `RIGDATA_20k.csv` / `RIGDATA_40k.csv` | `Time`, `Temperature(C)`, `Humidity(%)`, `Light(lux)`, `Occupancy`, `ENERGY` |

## Install And Run On Windows

Run these commands from this project folder:

```powershell
cd "C:\Users\HP\Downloads\testing2"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python test_prediction_api.py
```

The server listens on:

```text
http://127.0.0.1:5000
```

Open the dashboard:

```text
http://127.0.0.1:5000/
```

If port `5000` is busy:

```powershell
$env:PORT = "5050"
python test_prediction_api.py
```

Then open:

```text
http://127.0.0.1:5050/
```

## Testing With The Dashboard

1. Start the server with `python test_prediction_api.py`.
2. Open `http://127.0.0.1:5000/`.
3. Enter a date/time if you want the service to look for a matching historical CSV context.
4. Enter temperature, humidity, lux, and occupancy.
5. Occupancy is now a manual numeric input, so values such as `0`, `1`, `2`, `3`, or higher can be tested.
6. Click **Predict** to call `POST /predict`.
7. Click **Auto Next** to call `GET /predict_next`, which advances through the active CSV one row at a time.
8. The dashboard shows the active CSV beside the Auto Next button and displays energy values in `Wh`.

## Testing From The Terminal

Start the server first:

```powershell
python test_prediction_api.py
```

Check server metadata:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/metadata
```

Manual prediction:

```powershell
$body = @{
  temperature_c = 28.0
  humidity = 60.0
  lux = 400.0
  occupancy = 3
  datetime_str = "2026-05-24T14:20:00"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:5000/predict" `
  -Method Post `
  -Body $body `
  -ContentType "application/json"
```

Auto-next prediction from the active CSV:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/predict_next
```

Run the diagnostic script:

```powershell
python diagnose.py
```

Expected response shape:

```json
{
  "timestamp": "2026-05-24 14:20:00",
  "energy_unit": "Wh",
  "live_sensors": {
    "temperature_c": 28.0,
    "humidity": 60.0,
    "lux": 400.0,
    "occupancy": 3
  },
  "predictions": {
    "energy_unit": "Wh",
    "actual_energy_wh": null,
    "base_gru_wh": 4.8303,
    "lgbm_wh": 2.6878,
    "hybrid_final_wh": 3.7591,
    "safety_lower_bound_wh": 3.1952,
    "safety_upper_bound_wh": 4.3229,
    "hybrid_weight_gru": 0.5
  }
}
```

If an older client still expects the previous `*_kwh` or `*_kw` field names,
start the service with `INCLUDE_LEGACY_UNIT_ALIASES=1`. Those aliases carry the
same Wh values and should be treated as deprecated.

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Serves `test_dashboard.html`. |
| `GET` | `/metadata` | Shows active model directory, active CSV, row count, sequence length, and energy unit. |
| `POST` | `/predict` | Runs a manual prediction using user-provided sensor values. |
| `GET` | `/predict_next` | Advances one row through the active CSV and predicts. |

Manual request body:

```json
{
  "temperature_c": 28.0,
  "humidity": 60.0,
  "lux": 400.0,
  "occupancy": 3,
  "datetime_str": "2026-05-24T14:20:00"
}
```

`datetime_str` is optional. If supplied, the service tries to find a matching
CSV row so lag and rolling features come from a realistic historical context.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `5000` | HTTP server port. |
| `MODEL_ASSET_DIR` | `New folder` | Folder containing model/scaler artifacts and CSV files. |
| `CSV_FILE` | `datarig40.csv` | Friendly CSV filename inside `MODEL_ASSET_DIR`. |
| `CSV_PATH` | unset | Full explicit CSV path. Overrides `CSV_FILE`. |
| `INCLUDE_LEGACY_UNIT_ALIASES` | `0` | Set to `1` only if an older client still expects previous `*_kwh` or `*_kw` field names. |
| `MQTT_BROKER` | `localhost` | MQTT broker host. |
| `MQTT_PORT` | `1883` | MQTT broker port. |
| `PEAK_DEMAND_KW` | `2.4` | Kept for existing MQTT payload compatibility. |

## MQTT Integration

When the API starts, it launches an MQTT background bridge. It subscribes to:

```text
room/sensors
```

For every received sensor message, it runs the next CSV-context prediction and
publishes to:

```text
room/ml/predictions
```

New MQTT fields use Wh:

```json
{
  "predicted_energy_wh": 3.7591,
  "upper_bound_energy_wh": 4.3229,
  "lower_bound_energy_wh": 3.1952,
  "predicted_energy_range_wh": [3.1952, 4.3229],
  "energy_unit": "Wh"
}
```

If an existing MQTT consumer still reads old `*_kw` names, start the service
with `INCLUDE_LEGACY_UNIT_ALIASES=1` while you migrate that consumer to Wh.

## Raspberry Pi 5 Deployment

These steps assume Raspberry Pi OS on a Raspberry Pi 5 and a project path of:

```text
/home/pi/testing2
```

Copy the project folder to the Pi. Make sure this folder exists on the Pi:

```text
/home/pi/testing2/New folder
```

It must contain:

```text
datarig40.csv
RIGDATA_20k.csv
RIGDATA_40k.csv
lgb_model.joblib
scaler_lgb.joblib
scaler_gru.joblib
tegru_model.tflite
```

Install system packages:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git build-essential libgomp1
```

Create and activate a virtual environment:

```bash
cd /home/pi/testing2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Start the service:

```bash
python test_prediction_api.py
```

From another machine on the same network, open:

```text
http://<raspberry-pi-ip>:5000/
```

Find the Pi IP address:

```bash
hostname -I
```

Run with a specific CSV:

```bash
CSV_FILE=RIGDATA_40k.csv python test_prediction_api.py
```

Run on a different port:

```bash
PORT=5050 python test_prediction_api.py
```

### Raspberry Pi 5 Systemd Service

Create a service file:

```bash
sudo nano /etc/systemd/system/smart-grid-ml.service
```

Paste this:

```ini
[Unit]
Description=Smart Grid ML Prediction API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/testing2
Environment=PORT=5000
Environment=MODEL_ASSET_DIR=/home/pi/testing2/New folder
Environment=CSV_FILE=datarig40.csv
Environment=MQTT_BROKER=localhost
ExecStart=/home/pi/testing2/.venv/bin/python /home/pi/testing2/test_prediction_api.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable smart-grid-ml.service
sudo systemctl start smart-grid-ml.service
```

Check status and logs:

```bash
sudo systemctl status smart-grid-ml.service
journalctl -u smart-grid-ml.service -f
```

Restart after changing the CSV or code:

```bash
sudo systemctl restart smart-grid-ml.service
```

### Optional MQTT Broker On The Pi

If the Pi should host MQTT locally:

```bash
sudo apt install -y mosquitto mosquitto-clients
sudo systemctl enable mosquitto
sudo systemctl start mosquitto
```

Test publishing and subscribing:

```bash
mosquitto_sub -t room/ml/predictions
```

In another terminal:

```bash
mosquitto_pub -t room/sensors -m '{"temperature":28,"humidity":60,"lux":400,"occupancy":3}'
```

The current MQTT handler uses the next CSV context for prediction. If you want
MQTT payload values to override the current row exactly like the dashboard does,
that can be added as a small follow-up change.

## Troubleshooting

If the dashboard does not open:

```bash
curl http://127.0.0.1:5000/metadata
```

If the port is already in use:

```bash
PORT=5050 python test_prediction_api.py
```

If model loading fails, check that `New folder` contains all `.joblib` and
`.tflite` files listed above.

If LightGBM fails on the Raspberry Pi, verify OpenMP support is installed:

```bash
sudo apt install -y libgomp1
```

If the API starts but predictions fail with a CSV error, check:

```bash
python - <<'PY'
import pandas as pd
print(pd.read_csv("New folder/datarig40.csv").head())
PY
```

If you are testing from another computer and cannot reach the Pi, check:

```bash
hostname -I
sudo systemctl status smart-grid-ml.service
```

Then open:

```text
http://<raspberry-pi-ip>:5000/
```
