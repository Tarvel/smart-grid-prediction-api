"""
Smart Grid Hybrid AI Prediction Service
=======================================

FastAPI + MQTT service for the latest "New folder" model set:
  - TE-GRU TFLite model: New folder/tegru_model.tflite
  - LightGBM model: New folder/lgb_model.joblib
  - Feature scalers: New folder/scaler_gru.joblib and scaler_lgb.joblib

The HTTP and MQTT contracts are kept compatible with the older service.
"""

import json
import os
import threading
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from ai_edge_litert.interpreter import Interpreter

warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'.*",
    category=FutureWarning,
)

# Load .env from parent directory (PROJECT_CODE/.env)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
ENERGY_UNIT = "Wh"
INCLUDE_LEGACY_UNIT_ALIASES = os.environ.get("INCLUDE_LEGACY_UNIT_ALIASES", "0").lower() in {
    "1",
    "true",
    "yes",
}


def _resolve_path(path_value: str | Path, base: Path = BASE_DIR) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


MODEL_DIR = _resolve_path(os.environ.get("MODEL_ASSET_DIR", "New folder"))
CSV_OPTIONS = {
    "datarig40": "datarig40.csv",
    "datarig40.csv": "datarig40.csv",
    "rigdata_20k": "RIGDATA_20k.csv",
    "rigdata_20k.csv": "RIGDATA_20k.csv",
    "rigdata_40k": "RIGDATA_40k.csv",
    "rigdata_40k.csv": "RIGDATA_40k.csv",
}


def _resolve_csv_path() -> Path:
    if os.environ.get("CSV_PATH"):
        return _resolve_path(os.environ["CSV_PATH"])

    csv_file = os.environ.get("CSV_FILE", "datarig40.csv").strip()
    filename = CSV_OPTIONS.get(csv_file.lower(), csv_file)
    return _resolve_path(filename, MODEL_DIR)


CSV_PATH = _resolve_csv_path()

SEQ_LENGTH = 48
FEATURE_HISTORY = 24
WINDOW_SIZE = SEQ_LENGTH + FEATURE_HISTORY

# MQTT
MQTT_BROKER = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", 1883))
MQTT_CLIENT_ID = "ml-prediction-service"
TOPIC_SENSORS = "room/sensors"
TOPIC_ML_PREDICTIONS = "room/ml/predictions"

# Energy thresholds (for contract compatibility)
PEAK_DEMAND_KW = float(os.environ.get("PEAK_DEMAND_KW", 2.4))

# How often to run a prediction (seconds). Sensor messages arriving
# between intervals are ignored. Set via PREDICTION_INTERVAL_SECONDS in .env.
PREDICTION_INTERVAL_SECONDS = float(
    os.environ.get("PREDICTION_INTERVAL_SECONDS", 300)
)

# Simulation index (CSV row pointer)
current_sim_index = WINDOW_SIZE
_last_prediction_time: float = 0.0  # epoch seconds of last prediction


# =============================================================================
# LOAD AI ASSETS ON MODULE IMPORT
# =============================================================================
def _load_simulation_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {
        "timestamp": "Timestamp",
        "Time": "Timestamp",
        "temperature": "Temperature_C",
        "Temperature(C)": "Temperature_C",
        "humidity": "Humidity_%",
        "Humidity(%)": "Humidity_%",
        "lux": "Luminous_Intensity_Lux",
        "Light(lux)": "Luminous_Intensity_Lux",
        "occupancy": "Occupancy",
        "Occupancy": "Occupancy",
        "energy": "Energy_Wh",
        "ENERGY": "Energy_Wh",
        "Energy_kW": "Energy_Wh",
    }
    df = df.rename(columns={src: dst for src, dst in rename_map.items() if src in df.columns})

    required = ["Timestamp", "Temperature_C", "Humidity_%", "Luminous_Intensity_Lux", "Occupancy", "Energy_Wh"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["Energy_Wh"] = pd.to_numeric(df["Energy_Wh"], errors="coerce")

    if "time_of_day" in df.columns:
        df["time_of_day"] = pd.to_numeric(df["time_of_day"], errors="coerce")
        df["time_of_day"] = df["time_of_day"].fillna(df["Timestamp"].dt.hour)
    else:
        df["time_of_day"] = df["Timestamp"].dt.hour

    return df


print("Loading AI assets...")
df_sim = _load_simulation_csv(CSV_PATH)

scaler_lgb = joblib.load(MODEL_DIR / "scaler_lgb.joblib")
scaler_gru = joblib.load(MODEL_DIR / "scaler_gru.joblib")
lgb_model = joblib.load(MODEL_DIR / "lgb_model.joblib")

LGB_FEATURES = list(scaler_lgb.feature_names_in_)
GRU_FEATURES = list(scaler_gru.feature_names_in_)

interpreter = Interpreter(model_path=str(MODEL_DIR / "tegru_model.tflite"))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"[OK] AI assets loaded - CSV: {CSV_PATH.name}, unit: {ENERGY_UNIT}")


def _find_csv_index(datetime_str: str) -> int | None:
    """Find a CSV row that can provide matching historical context."""
    try:
        target = pd.Timestamp(datetime_str)
    except Exception:
        return None

    valid_df = df_sim[df_sim.index >= WINDOW_SIZE]
    if len(valid_df) == 0:
        return WINDOW_SIZE

    diffs = (valid_df["Timestamp"] - target).abs()
    best_idx = int(diffs.idxmin())
    if diffs.loc[best_idx] <= pd.Timedelta(hours=1):
        return best_idx

    mask_mdh = (
        (valid_df["Timestamp"].dt.month == target.month)
        & (valid_df["Timestamp"].dt.dayofweek == target.dayofweek)
        & (valid_df["Timestamp"].dt.hour == target.hour)
    )
    matches_mdh = valid_df[mask_mdh]
    if len(matches_mdh) > 0:
        return int(matches_mdh.index[-1])

    mask_dh = (
        (valid_df["Timestamp"].dt.dayofweek == target.dayofweek)
        & (valid_df["Timestamp"].dt.hour == target.hour)
    )
    matches_dh = valid_df[mask_dh]
    if len(matches_dh) > 0:
        return int(matches_dh.index[-1])

    mask_h = valid_df["Timestamp"].dt.hour == target.hour
    matches_h = valid_df[mask_h]
    if len(matches_h) > 0:
        return int(matches_h.index[-1])

    return WINDOW_SIZE


def _context_window(ctx_index: int) -> pd.DataFrame:
    ctx_index = max(WINDOW_SIZE, min(int(ctx_index), len(df_sim) - 1))
    return df_sim.iloc[ctx_index - WINDOW_SIZE : ctx_index + 1].copy()


# =============================================================================
# BAYESIAN UNCERTAINTY AND ADAPTIVE WEIGHT ESTIMATOR
# =============================================================================
class MHWeightEstimator:
    """Metropolis-Hastings sampler for adaptive component weighting."""

    def __init__(self, n_iterations=1000, proposal_std=0.02, temperature=1e-4):
        self.n_iterations = n_iterations
        self.proposal_std = proposal_std
        self.temperature = temperature

    def estimate_weight(self, y_true: np.ndarray, pred_gru: np.ndarray, pred_lgbm: np.ndarray, w_init=0.5) -> float:
        if len(y_true) < 2:
            return w_init

        def hybrid_loss(w):
            blended = w * pred_gru + (1 - w) * pred_lgbm
            return float(np.mean((y_true - blended) ** 2))

        w_current = w_init
        loss_current = hybrid_loss(w_current)
        best_w = w_current
        best_loss = loss_current

        rng = np.random.RandomState(int(abs(loss_current * 1e5)) % (2**31))

        for _ in range(self.n_iterations):
            w_proposed = np.clip(w_current + rng.normal(0, self.proposal_std), 0.0, 1.0)
            loss_proposed = hybrid_loss(w_proposed)

            delta_loss = loss_proposed - loss_current
            if delta_loss <= 0:
                alpha = 1.0
            else:
                alpha = float(np.exp(-delta_loss / max(self.temperature, 1e-8)))

            if rng.uniform(0, 1) < alpha:
                w_current = w_proposed
                loss_current = loss_proposed
                if loss_current < best_loss:
                    best_loss = loss_current
                    best_w = w_current

        return best_w


class HistoryTracker:
    """Rolling window of true values and component predictions."""

    def __init__(self, max_size=200):
        self.y_true = []
        self.pred_gru = []
        self.pred_lgbm = []
        self.max_size = max_size

    def add(self, actual, gru_val, lgbm_val):
        if actual is not None and not np.isnan(actual):
            self.y_true.append(actual)
            self.pred_gru.append(gru_val)
            self.pred_lgbm.append(lgbm_val)
            if len(self.y_true) > self.max_size:
                self.y_true.pop(0)
                self.pred_gru.pop(0)
                self.pred_lgbm.pop(0)

    def clear(self):
        self.y_true.clear()
        self.pred_gru.clear()
        self.pred_lgbm.clear()

    def get(self):
        if len(self.y_true) >= 3:
            return np.array(self.y_true), np.array(self.pred_gru), np.array(self.pred_lgbm)
        return None, None, None


mh_estimator = MHWeightEstimator()
history_tracker = HistoryTracker()


# =============================================================================
# CORE PREDICTION PIPELINE
# =============================================================================
def _build_feature_frame(live_window: pd.DataFrame) -> pd.DataFrame:
    """Recreate the feature engineering from New folder/MODEL.ipynb."""
    w = live_window.copy()
    w["Timestamp"] = pd.to_datetime(w["Timestamp"])
    w = w.sort_values("Timestamp")

    w["temperature"] = pd.to_numeric(w["Temperature_C"], errors="coerce")
    w["humidity"] = pd.to_numeric(w["Humidity_%"], errors="coerce")
    w["lux"] = pd.to_numeric(w["Luminous_Intensity_Lux"], errors="coerce")
    w["occupancy"] = pd.to_numeric(w["Occupancy"], errors="coerce")

    w["hour"] = w["Timestamp"].dt.hour
    w["day_of_week"] = w["Timestamp"].dt.dayofweek
    if "time_of_day" in w.columns:
        w["time_of_day"] = pd.to_numeric(w["time_of_day"], errors="coerce")
        w["time_of_day"] = w["time_of_day"].fillna(w["hour"])
    else:
        w["time_of_day"] = w["hour"]

    w["hour_sin"] = np.sin((2 * np.pi * w["hour"]) / 24)
    w["hour_cos"] = np.cos((2 * np.pi * w["hour"]) / 24)
    w["dow_sin"] = np.sin((2 * np.pi * w["day_of_week"]) / 7)
    w["dow_cos"] = np.cos((2 * np.pi * w["day_of_week"]) / 7)
    w["is_weekend"] = w["day_of_week"].isin([5, 6]).astype(int)

    for col in ["temperature", "humidity", "lux"]:
        w[f"{col}_lag1"] = w[col].shift(1)
        w[f"{col}_lag24"] = w[col].shift(24)
        w[f"{col}_mean3"] = w[col].rolling(3).mean()
        w[f"{col}_mean24"] = w[col].rolling(24).mean()

    feature_columns = list(dict.fromkeys(LGB_FEATURES + GRU_FEATURES + ["hour"]))
    features = w[feature_columns].copy()
    model_columns = list(dict.fromkeys(LGB_FEATURES + GRU_FEATURES))
    return features.dropna(subset=model_columns)


def _invoke_gru(gru_sequence: np.ndarray, target_hour: float) -> float:
    numeric_input = np.array([gru_sequence], dtype=np.float32)
    hour_input = np.array([[target_hour]], dtype=np.float32)

    for detail in input_details:
        name = detail["name"].lower()
        shape = tuple(int(dim) for dim in detail["shape"])
        value = hour_input if "hour" in name or (len(shape) == 2 and shape[-1] == 1) else numeric_input
        interpreter.set_tensor(detail["index"], value.astype(detail["dtype"]))

    interpreter.invoke()
    return float(np.ravel(interpreter.get_tensor(output_details[0]["index"]))[0])


def run_prediction(live_window: pd.DataFrame) -> dict:
    """Run TE-GRU + LightGBM hybrid prediction on a context window."""
    current_hour_data = live_window.iloc[-1].copy()
    feature_df = _build_feature_frame(live_window)

    if len(feature_df) < SEQ_LENGTH + 1:
        raise ValueError(
            f"Not enough engineered rows for inference. Need {SEQ_LENGTH + 1}, got {len(feature_df)}."
        )

    lgb_input = scaler_lgb.transform(feature_df[LGB_FEATURES].iloc[[-1]])
    scaled_gru = scaler_gru.transform(feature_df[GRU_FEATURES])

    gru_sequence = scaled_gru[-(SEQ_LENGTH + 1) : -1]
    target_hour = float(feature_df["hour"].iloc[-1])

    gru_raw = _invoke_gru(gru_sequence, target_hour)
    lgbm_raw = float(lgb_model.predict(lgb_input)[0])

    y_hist, gru_hist, lgbm_hist = history_tracker.get()
    if y_hist is not None:
        best_w = mh_estimator.estimate_weight(y_hist, gru_hist, lgbm_hist)
        blended_hist = best_w * gru_hist + (1 - best_w) * lgbm_hist
        residual_std = float(np.std(y_hist - blended_hist))
    else:
        best_w = 0.5
        residual_std = None

    hybrid_final_wh = best_w * gru_raw + (1 - best_w) * lgbm_raw
    if residual_std is None:
        residual_std = max(0.05, abs(hybrid_final_wh) * 0.10)

    z = 1.5
    lower_bound = max(0.0, hybrid_final_wh - z * residual_std)
    upper_bound = hybrid_final_wh + z * residual_std

    actual_val = current_hour_data.get("Energy_Wh", np.nan)
    actual_wh = round(float(actual_val), 4) if pd.notna(actual_val) else None

    history_tracker.add(actual_wh, gru_raw, lgbm_raw)

    base_gru_wh = round(gru_raw, 4)
    lgbm_wh = round(lgbm_raw, 4)
    hybrid_final_wh = round(hybrid_final_wh, 4)
    lower_bound = round(lower_bound, 4)
    upper_bound = round(upper_bound, 4)

    predictions = {
        "energy_unit": ENERGY_UNIT,
        "actual_energy_wh": actual_wh,
        "base_gru_wh": base_gru_wh,
        "lgbm_wh": lgbm_wh,
        "hybrid_final_wh": hybrid_final_wh,
        "safety_lower_bound_wh": lower_bound,
        "safety_upper_bound_wh": upper_bound,
        "hybrid_weight_gru": round(best_w, 4),
    }
    if INCLUDE_LEGACY_UNIT_ALIASES:
        predictions.update(
            {
                "actual_energy_kw": actual_wh,
                "base_gru_kwh": base_gru_wh,
                "lgbm_kwh": lgbm_wh,
                "hybrid_final_kwh": hybrid_final_wh,
                "safety_lower_bound": lower_bound,
                "safety_upper_bound": upper_bound,
            }
        )

    return {
        "timestamp": str(current_hour_data["Timestamp"]),
        "energy_unit": ENERGY_UNIT,
        "live_sensors": {
            "temperature_c": float(current_hour_data["Temperature_C"]),
            "humidity": float(current_hour_data["Humidity_%"]),
            "lux": float(current_hour_data["Luminous_Intensity_Lux"]),
            "occupancy": int(current_hour_data["Occupancy"]),
        },
        "predictions": predictions,
    }


# =============================================================================
# MQTT BRIDGE
# =============================================================================
mqtt_client = mqtt.Client(
    client_id=MQTT_CLIENT_ID,
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
)


def build_mqtt_payload(result: dict) -> dict:
    """Convert internal result into the flat MQTT contract payload."""
    pred = result["predictions"]
    predicted_wh = pred["hybrid_final_wh"]
    upper_wh = pred["safety_upper_bound_wh"]
    lower_wh = pred["safety_lower_bound_wh"]
    payload = {
        "predicted_energy_wh": predicted_wh,
        "upper_bound_energy_wh": upper_wh,
        "lower_bound_energy_wh": lower_wh,
        "predicted_energy_range_wh": [lower_wh, upper_wh],
        "energy_unit": ENERGY_UNIT,
        "peak_demand": PEAK_DEMAND_KW,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "fastapi-local-model",
    }
    if INCLUDE_LEGACY_UNIT_ALIASES:
        payload.update(
            {
                "predicted_energy_kw": predicted_wh,
                "upper_bound_energy_kw": upper_wh,
                "lower_bound_energy_kw": lower_wh,
                "predicted_energy_range": [lower_wh, upper_wh],
            }
        )
    return payload


def on_mqtt_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        client.subscribe(TOPIC_SENSORS, qos=1)
        print(f"[OK] MQTT bridge connected - subscribed to '{TOPIC_SENSORS}'")
    else:
        print(f"[ERROR] MQTT connection failed (rc={rc})")


def on_mqtt_message(client, userdata, msg):
    """Sensor message arrives — run prediction if interval has elapsed."""
    global current_sim_index, _last_prediction_time

    import time as _time

    # ── Throttle: skip if interval hasn't elapsed ──
    now = _time.time()
    if now - _last_prediction_time < PREDICTION_INTERVAL_SECONDS:
        return  # too soon, wait for next interval
    _last_prediction_time = now

    try:
        # Parse the sensor payload to extract timestamp
        sensor_data = json.loads(msg.payload.decode("utf-8"))
        sensor_ts = sensor_data.get("timestamp")

        # Try to match CSV context to the sensor timestamp
        matched_idx = _find_csv_index(sensor_ts) if sensor_ts else None
        if matched_idx is not None:
            ctx_index = matched_idx
        else:
            if current_sim_index >= len(df_sim):
                print("[WARN] Simulation finished - resetting index")
                current_sim_index = WINDOW_SIZE
            ctx_index = current_sim_index
            current_sim_index += 1

        live_window = _context_window(ctx_index)
        result = run_prediction(live_window)
        mqtt_payload = build_mqtt_payload(result)

        client.publish(TOPIC_ML_PREDICTIONS, json.dumps(mqtt_payload), qos=1)
        print(
            f"  -> MQTT prediction ({PREDICTION_INTERVAL_SECONDS:.0f}s interval): "
            f"{mqtt_payload['predicted_energy_wh']:.4f} {ENERGY_UNIT}"
            f"  [ts={sensor_ts or 'n/a'}]"
        )
    except Exception as exc:
        print(f"[ERROR] MQTT prediction error: {exc}")


def start_mqtt_bridge():
    """Connect MQTT client and start the network loop in the background."""
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)

    def _try_connect():
        while True:
            try:
                mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
                mqtt_client.loop_start()
                print(f"MQTT bridge started -> {MQTT_BROKER}:{MQTT_PORT}")
                return
            except Exception as exc:
                print(f"[WARN] MQTT broker unreachable ({exc}) - retrying in 5s...")
                import time

                time.sleep(5)

    threading.Thread(target=_try_connect, daemon=True).start()


# =============================================================================
# FASTAPI APP
# =============================================================================
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Start MQTT bridge on boot, clean up on shutdown."""
    start_mqtt_bridge()
    yield
    try:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("MQTT bridge disconnected.")
    except Exception:
        pass


app = FastAPI(
    title="Smart Grid Hybrid AI - TEST SIMULATOR",
    description="HTTP endpoints are for manual testing. Production data flows through MQTT.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SensorInput(BaseModel):
    """Sensor values typed by the user for testing."""

    temperature_c: float = 28.0
    humidity: float = 60.0
    lux: float = 400.0
    occupancy: int = 1
    datetime_str: str | None = None


@app.post("/predict")
def predict_manual(sensor: SensorInput):
    """Accept manual sensor values, inject them into context, and predict."""
    global current_sim_index

    matched_idx = _find_csv_index(sensor.datetime_str) if sensor.datetime_str else None
    if matched_idx is not None:
        ctx_index = matched_idx
    else:
        if current_sim_index >= len(df_sim):
            current_sim_index = WINDOW_SIZE
        ctx_index = current_sim_index

    live_window = _context_window(ctx_index)

    idx = live_window.index[-1]
    live_window.loc[idx, "Temperature_C"] = sensor.temperature_c
    live_window.loc[idx, "Humidity_%"] = sensor.humidity
    live_window.loc[idx, "Luminous_Intensity_Lux"] = sensor.lux
    live_window.loc[idx, "Occupancy"] = sensor.occupancy
    live_window.loc[idx, "Energy_Wh"] = np.nan

    if sensor.datetime_str:
        try:
            user_ts = pd.Timestamp(sensor.datetime_str)
        except Exception:
            user_ts = live_window.loc[idx, "Timestamp"]
    else:
        user_ts = live_window.loc[idx, "Timestamp"]

    live_window.loc[idx, "Timestamp"] = user_ts
    live_window.loc[idx, "time_of_day"] = user_ts.hour

    return run_prediction(live_window)


@app.get("/metadata")
def metadata():
    """Return active model, CSV, and unit metadata for dashboards and testers."""
    return {
        "model_dir": str(MODEL_DIR),
        "csv_path": str(CSV_PATH),
        "csv_name": CSV_PATH.name,
        "available_csv_files": sorted(set(CSV_OPTIONS.values())),
        "energy_unit": ENERGY_UNIT,
        "include_legacy_unit_aliases": INCLUDE_LEGACY_UNIT_ALIASES,
        "sequence_length": SEQ_LENGTH,
        "window_size": WINDOW_SIZE,
        "rows_loaded": len(df_sim),
    }


@app.get("/predict_next")
def predict_next_hour():
    """Advance one row through the CSV dataset and return the prediction."""
    global current_sim_index

    if current_sim_index >= len(df_sim):
        return {"error": "Simulation finished. End of dataset."}

    live_window = _context_window(current_sim_index)
    current_sim_index += 1

    return run_prediction(live_window)


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    html_path = BASE_DIR / "test_dashboard.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>test_dashboard.html not found.</h1>"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
