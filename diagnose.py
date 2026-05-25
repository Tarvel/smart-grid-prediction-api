"""Diagnostic run for the latest model set in New folder."""

import sys

import numpy as np

from new_prediction_api import WINDOW_SIZE, df_sim, history_tracker, run_prediction


sys.stdout.reconfigure(encoding="utf-8")


def _metrics(actuals: np.ndarray, preds: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(actuals - preds)))
    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
    ss_res = float(np.sum((actuals - preds) ** 2))
    ss_tot = float(np.sum((actuals - np.mean(actuals)) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1 - ss_res / ss_tot
    return {"r2": r2, "mae": mae, "rmse": rmse}


def _predict_row(idx: int):
    window = df_sim.iloc[idx - WINDOW_SIZE : idx + 1].copy()
    result = run_prediction(window)
    pred = result["predictions"]
    return (
        pred["actual_energy_wh"],
        pred["base_gru_wh"],
        pred["lgbm_wh"],
        pred["hybrid_final_wh"],
    )


def _segment_ranges():
    segment_size = min(200, max(25, (len(df_sim) - WINDOW_SIZE) // 10))
    starts = [
        ("Early", WINDOW_SIZE),
        ("Middle", max(WINDOW_SIZE, len(df_sim) // 2)),
        ("Late", max(WINDOW_SIZE, int(len(df_sim) * 0.8))),
        ("End", max(WINDOW_SIZE, len(df_sim) - segment_size - 1)),
    ]

    for name, start in starts:
        end = min(start + segment_size, len(df_sim))
        if start < end:
            yield name, range(start, end)


for name, rows in _segment_ranges():
    history_tracker.clear()
    actuals, grus, lgbms, hybrids = [], [], [], []

    for idx in rows:
        actual, gru, lgbm, hybrid = _predict_row(idx)
        if actual is None:
            continue
        actuals.append(actual)
        grus.append(gru)
        lgbms.append(lgbm)
        hybrids.append(hybrid)

    actuals = np.array(actuals)
    grus = np.array(grus)
    lgbms = np.array(lgbms)
    hybrids = np.array(hybrids)

    print(f"\n{'=' * 60}")
    print(f"{name} rows ({len(actuals)} predictions)")
    print(f"Actual range: [{actuals.min():.2f}, {actuals.max():.2f}]")

    for label, preds in [("GRU", grus), ("LightGBM", lgbms), ("Hybrid", hybrids)]:
        metric = _metrics(actuals, preds)
        print(
            f"{label:8s} R2={metric['r2']:.4f} "
            f"MAE={metric['mae']:.4f} RMSE={metric['rmse']:.4f}"
        )
