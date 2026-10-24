"""A small expected-goals model fitted on the shots in this repository.

The event data has no xG, so we fit a logistic regression on shot location
(distance, visible goal angle) plus shot type flags. Penalties get a fixed
value. Coefficients live in ``analysis/xg_model.json``; refit with
``python -m analysis.fit_xg`` when many new matches have been added.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from .data import HALF_LEN, POST_Y, REPO, load_events, match_index

MODEL_PATH = REPO / "analysis" / "xg_model.json"
PENALTY_XG = 0.76
FEATURES = ["const", "log_dist", "angle", "header", "direct_fk", "open_goal", "one_v_one"]


def shot_features(shots: pd.DataFrame) -> pd.DataFrame:
    dx = (HALF_LEN - shots.x).clip(lower=0.5)
    dy = shots.y.abs()
    dist = np.hypot(dx, dy)
    # Angle subtended by the goal mouth from the shot location.
    angle = np.arctan2(2 * POST_Y * dx, dx**2 + dy**2 - POST_Y**2)
    angle = np.where(angle < 0, angle + np.pi, angle)
    return pd.DataFrame({
        "const": 1.0,
        "log_dist": np.log(dist),
        "angle": angle,
        "header": ((shots.body_part == "HEAD") | (shots.action == "HEADER")).astype(float),
        "direct_fk": (shots.action == "DIRECT_FREE_KICK").astype(float),
        "open_goal": (shots.action == "OPEN_GOAL_SHOT").astype(float),
        "one_v_one": (shots.action == "ONE_VS_ONE_AGAINST_GK").astype(float),
    }, index=shots.index)


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    reg = l2 * np.eye(X.shape[1])
    reg[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        grad = X.T @ (y - p) - reg @ beta
        hess = X.T @ (X * (p * (1 - p))[:, None]) + reg
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def fit() -> dict:
    shots = []
    for m in match_index():
        df = load_events(m)
        shots.append(df[(df.action_type == "SHOT") & (df.action != "PENALTY_KICK") & df.x.notna()])
    shots = pd.concat(shots, ignore_index=True)
    X = shot_features(shots)[FEATURES].to_numpy()
    y = (shots.result == "SUCCESS").to_numpy(float)
    beta = _fit_logistic(X, y)
    p = 1 / (1 + np.exp(-X @ beta))
    model = {
        "coefficients": dict(zip(FEATURES, beta.round(5).tolist())),
        "penalty_xg": PENALTY_XG,
        "n_shots": int(len(y)),
        "n_goals": int(y.sum()),
        "brier": round(float(np.mean((p - y) ** 2)), 5),
        "fitted": date.today().isoformat(),
    }
    MODEL_PATH.write_text(json.dumps(model, indent=2))
    return model


def _model() -> dict:
    if not MODEL_PATH.exists():
        return fit()
    return json.loads(MODEL_PATH.read_text())


def add_xg(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``xg`` column (NaN for non-shots)."""
    model = _model()
    beta = np.array([model["coefficients"][f] for f in FEATURES])
    df = df.copy()
    df["xg"] = np.nan
    shots = df[(df.action_type == "SHOT") & df.x.notna()]
    if len(shots):
        X = shot_features(shots)[FEATURES].to_numpy()
        xg = 1 / (1 + np.exp(-X @ beta))
        xg = np.where(shots.action == "PENALTY_KICK", model["penalty_xg"], xg)
        df.loc[shots.index, "xg"] = xg
    return df
