from __future__ import annotations

import math
from statistics import mean
from typing import Dict, List, Optional, Tuple


def drift_forecast(history: List[float]) -> float:
    if not history:
        return 0.0
    mu = mean(history)
    if len(history) < 2:
        return mu
    diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
    drift = mean(diffs)
    return mu + drift


def ar1_forecast(history: List[float]) -> float:
    if not history:
        return 0.0
    if len(history) < 2:
        return history[-1]

    x = history[:-1]
    y = history[1:]
    x_mean = mean(x)
    y_mean = mean(y)
    var_x = sum((v - x_mean) ** 2 for v in x)
    if var_x <= 1e-12:
        a = 0.0
        b = y_mean
    else:
        cov_xy = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
        a = cov_xy / var_x
        b = y_mean - a * x_mean
    return a * history[-1] + b


def forecast_with_method(history: List[float], method: str) -> float:
    if method == "drift":
        return drift_forecast(history)
    if method == "ar1":
        return ar1_forecast(history)
    raise ValueError(f"Unsupported baseline method: {method}")


def rolling_residuals(values: List[float], window: int, method: str) -> Tuple[List[float], List[float]]:
    z_hat: List[float] = [math.nan] * len(values)
    residuals: List[float] = [math.nan] * len(values)
    for idx in range(len(values)):
        start = idx - window
        if start < 0:
            continue
        history = values[start:idx]
        if len(history) < window:
            continue
        pred = forecast_with_method(history, method)
        z_hat[idx] = pred
        residuals[idx] = values[idx] - pred
    return z_hat, residuals


def compute_all_itemset_residuals(
    z_series: Dict[Tuple[int, ...], List[float]],
    window: int,
    method: str,
) -> Tuple[Dict[Tuple[int, ...], List[float]], Dict[Tuple[int, ...], List[float]]]:
    z_hat_by_itemset: Dict[Tuple[int, ...], List[float]] = {}
    residual_by_itemset: Dict[Tuple[int, ...], List[float]] = {}
    for itemset, values in z_series.items():
        z_hat, residual = rolling_residuals(values, window=window, method=method)
        z_hat_by_itemset[itemset] = z_hat
        residual_by_itemset[itemset] = residual
    return z_hat_by_itemset, residual_by_itemset


def yearly_percentile_threshold(values: List[float], percentile: float) -> Optional[float]:
    finite = sorted(v for v in values if not math.isnan(v))
    if not finite:
        return None
    idx = max(0, min(len(finite) - 1, int((percentile / 100.0) * (len(finite) - 1))))
    return finite[idx]
