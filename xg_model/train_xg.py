"""Train and evaluate shot value (xG) models on the shots table from build_dataset.py.

Three models are compared with match-grouped 5-fold cross-validation:
  1. location   - logistic regression on shot location only
  2. full_logit - logistic regression on all pre-shot features
  3. full_gbm   - LightGBM on all pre-shot features

Post-shot information (targetPoint, keeper dive, woodwork, end location) is never
used, so the values describe chance quality before the ball is struck.
Penalties are excluded from training and get a fixed xG of PENALTY_XG.
"""
import os
import re
import warnings

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
PENALTY_XG = 0.76
SEED = 42

LOCATION_NUM = ["distance", "goal_mouth_angle", "x", "abs_y"]
FULL_NUM = LOCATION_NUM + [
    "angle_to_centre", "in_box", "in_six_yard_box", "from_set_piece",
    "pressure", "opponents", "dist_to_opponent",
    "gk_dist_from_line", "gk_dist_to_shooter", "gk_off_centre",
    "assist_distance", "time_since_assist",
]
FULL_CAT = ["shot_type", "body_part", "phase", "prev_action", "assist_type"]
RARE_MIN = 20


def load():
    df = pd.read_csv(os.path.join(OUT_DIR, "shots.csv"))
    for c in FULL_CAT:
        counts = df[c].value_counts()
        df[c] = df[c].where(df[c].map(counts) >= RARE_MIN, "OTHER")
    df["has_assist"] = df["assist_distance"].notna().astype(int)
    for c in ["assist_distance", "time_since_assist"]:
        df[c] = df[c].fillna(-1)
    return df


def squad_names(df):
    """Map squad_id -> team name using the "<home>_vs_<away>_<id>.json" filenames."""
    candidates = {}
    for f, g in df.groupby("match_file"):
        m = re.match(r"\d{4}-\d\d-\d\d_(.+)_vs_(.+)_\d+\.json", f)
        for team in m.groups():
            ids = set(g.squad_id)
            candidates[team] = candidates.get(team, ids) & ids
    return {next(iter(ids)): team.replace("_", " ") for team, ids in candidates.items() if len(ids) == 1}


def location_model():
    pre = ColumnTransformer([
        ("spl", SplineTransformer(n_knots=5, degree=3), ["distance", "goal_mouth_angle"]),
        ("num", StandardScaler(), LOCATION_NUM),
    ])
    return make_pipeline(pre, LogisticRegression(C=1.0, max_iter=2000))


def full_logit():
    pre = ColumnTransformer([
        ("spl", SplineTransformer(n_knots=5, degree=3), ["distance", "goal_mouth_angle"]),
        ("num", StandardScaler(), FULL_NUM + ["has_assist"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FULL_CAT),
    ])
    return make_pipeline(pre, LogisticRegression(C=0.3, max_iter=4000))


class GBM:
    """LightGBM wrapper with a sklearn-like fit/predict_proba on a DataFrame."""

    params = dict(
        objective="binary", learning_rate=0.03, n_estimators=400, num_leaves=8,
        min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=5.0, random_state=SEED, verbose=-1,
    )

    def _x(self, df):
        X = df[FULL_NUM + ["has_assist"] + FULL_CAT].copy()
        for c in FULL_CAT:
            X[c] = pd.Categorical(X[c], categories=self.cats_[c])
        return X

    def fit(self, df, y):
        self.cats_ = {c: sorted(df[c].unique()) for c in FULL_CAT}
        self.model_ = lgb.LGBMClassifier(**self.params).fit(self._x(df), y)
        return self

    def predict_proba(self, df):
        return self.model_.predict_proba(self._x(df))


MODELS = {"location": location_model, "full_logit": full_logit, "full_gbm": GBM}


def metrics(y, p):
    return {
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "auc": roc_auc_score(y, p),
        "xg_total": p.sum(),
        "goals": int(y.sum()),
    }


def cross_validate(train):
    y = train["goal"].values
    groups = train["match_id"].values
    oof = {name: np.zeros(len(train)) for name in MODELS}
    for tr, te in GroupKFold(n_splits=5).split(train, y, groups):
        for name, make in MODELS.items():
            m = make().fit(train.iloc[tr], y[tr])
            oof[name][te] = m.predict_proba(train.iloc[te])[:, 1]
    rows = {name: metrics(y, p) for name, p in oof.items()}
    base = np.full(len(y), y.mean())
    rows["baseline_mean"] = metrics(y, base)
    pxt = train["provider_pxT"].values
    rows["provider_pxT (AUC only)"] = {"auc": roc_auc_score(y, pxt)}
    return oof, pd.DataFrame(rows).T


def plot_calibration(y, oof, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    for name, p in oof.items():
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=name)
    ax.set_xlabel("Predicted xG")
    ax.set_ylabel("Observed goal rate")
    ax.set_title("Calibration (out-of-fold, decile bins)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_location_surface(model, path):
    from mplsoccer import VerticalPitch

    xs = np.linspace(52.5 - 35, 52.5 - 0.5, 70)
    ys = np.linspace(-34, 34, 136)
    gx, gy = np.meshgrid(xs, ys)
    grid = pd.DataFrame({"x": gx.ravel(), "y": gy.ravel()})
    grid["abs_y"] = grid.y.abs()
    grid["distance"] = np.hypot(52.5 - grid.x, grid.y)
    dx = 52.5 - grid.x
    grid["goal_mouth_angle"] = np.degrees(np.abs(np.arctan2(3.66 - grid.y, dx) - np.arctan2(-3.66 - grid.y, dx)))
    z = model.predict_proba(grid)[:, 1].reshape(gx.shape)

    pitch = VerticalPitch(pitch_type="custom", pitch_length=105, pitch_width=68, half=True, line_zorder=2)
    fig, ax = pitch.draw(figsize=(7, 6))
    # mplsoccer custom pitch runs 0..105 / 0..68; shift from centred coords
    mesh = ax.pcolormesh(gy + 34, gx + 52.5, z, cmap="viridis", shading="auto", vmin=0, vmax=0.6, zorder=1)
    fig.colorbar(mesh, ax=ax, shrink=0.7, label="xG (location model)")
    ax.set_title("Shot value by location")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_importance(gbm, path):
    imp = pd.Series(gbm.model_.booster_.feature_importance("gain"), index=gbm.model_.feature_name_)
    imp = (imp / imp.sum()).sort_values()
    fig, ax = plt.subplots(figsize=(6, 6))
    imp.plot.barh(ax=ax)
    ax.set_xlabel("Share of total gain")
    ax.set_title("LightGBM feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    df = load()
    is_pen = df["shot_type"] == "PENALTY_KICK"
    train = df[~is_pen].reset_index(drop=True)

    oof, scores = cross_validate(train)
    print(scores.round(4).to_string())
    scores.to_csv(os.path.join(OUT_DIR, "model_scores.csv"))

    cv_scores = scores.loc[list(MODELS)]
    best = cv_scores["log_loss"].idxmin()
    print(f"\nBest model by log loss: {best}")

    # Final models on all non-penalty shots
    y = train["goal"].values
    final = {name: make().fit(train, y) for name, make in MODELS.items()}
    joblib.dump({"models": final, "best": best, "penalty_xg": PENALTY_XG,
                 "rare_min": RARE_MIN}, os.path.join(OUT_DIR, "xg_models.joblib"))

    # Shot table with honest out-of-fold xG
    train["xg_location"] = oof["location"]
    train["xg_full_logit"] = oof["full_logit"]
    train["xg_full_gbm"] = oof["full_gbm"]
    pens = df[is_pen].copy()
    for c in ["xg_location", "xg_full_logit", "xg_full_gbm"]:
        pens[c] = PENALTY_XG
    shots = pd.concat([train, pens], ignore_index=True)
    shots["xg"] = shots[f"xg_{best}"]
    shots["team"] = shots["squad_id"].map(squad_names(shots))
    shots.sort_values(["league", "match_file", "period", "game_time_sec"]).to_csv(
        os.path.join(OUT_DIR, "shots_with_xg.csv"), index=False)

    plot_calibration(y, oof, os.path.join(OUT_DIR, "calibration.png"))
    plot_location_surface(final["location"], os.path.join(OUT_DIR, "xg_by_location.png"))
    plot_importance(final["full_gbm"], os.path.join(OUT_DIR, "feature_importance.png"))

    ere = shots[shots.league.str.startswith("Eredivisie")]
    players = (ere.groupby(["player_id", "player_name"])
               .agg(shots=("goal", "size"), goals=("goal", "sum"), xg=("xg", "sum"),
                    npxg=("xg", lambda s: s[ere.loc[s.index, "shot_type"] != "PENALTY_KICK"].sum()))
               .assign(xg_per_shot=lambda d: d.xg / d.shots, goals_minus_xg=lambda d: d.goals - d.xg)
               .sort_values("xg", ascending=False).round(3))
    players.to_csv(os.path.join(OUT_DIR, "eredivisie_players_xg.csv"))
    print("\nEredivisie top 15 by xG:\n", players.head(15).to_string())

    teams = (ere.groupby(["squad_id", "team"])
             .agg(matches=("match_id", "nunique"), shots=("goal", "size"), goals=("goal", "sum"), xg=("xg", "sum"))
             .assign(xg_per_match=lambda d: d.xg / d.matches, goals_minus_xg=lambda d: d.goals - d.xg)
             .sort_values("xg", ascending=False).round(3))
    teams.to_csv(os.path.join(OUT_DIR, "eredivisie_teams_xg.csv"))
    print("\nEredivisie teams:\n", teams.to_string())


if __name__ == "__main__":
    main()
