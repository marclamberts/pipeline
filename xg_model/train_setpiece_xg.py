"""Set-piece shot value (xG) model.

Uses only shots that come from a set piece (corner, free kick, direct free kick,
throw-in) plus all penalties, and adds set-piece features to the general ones:
restart type and location, first contact vs. later phase, time since the restart,
delivery length and in/out-swing proxy.

Compared with match-grouped 5-fold CV against the general model's out-of-fold xG
on the same shots (run train_xg.py first). With ~1,900 set-piece shots the general
model, which learns from all ~6,000 shots, scores better, so its values are used as
`xg` whenever it wins; the set-piece features are then used for the breakdowns.
"""
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from train_xg import (FULL_CAT, FULL_NUM, LOCATION_NUM, OUT_DIR, PENALTY_XG, RARE_MIN, SEED,
                      location_model, metrics, plot_calibration, squad_names)

SP_DIR = os.path.join(OUT_DIR, "set_pieces")

SP_NUM = [c for c in FULL_NUM if c != "from_set_piece"] + [
    "restart_x", "restart_abs_y", "restart_dist_to_goal", "time_since_restart",
    "events_since_restart", "first_contact", "first_phase", "delivery_distance",
    "delivery_swing", "short_delivery",
]
SP_CAT = [c for c in FULL_CAT if c != "phase"] + ["restart_type"]


def load():
    df = pd.read_csv(os.path.join(OUT_DIR, "shots.csv"))
    df = df[df.is_set_piece == 1].reset_index(drop=True)
    for c in SP_CAT:
        counts = df[c].value_counts()
        df[c] = df[c].where(df[c].map(counts) >= RARE_MIN, "OTHER")
    df["has_assist"] = df["assist_distance"].notna().astype(int)
    df["has_delivery"] = df["delivery_distance"].notna().astype(int)
    for c in ["assist_distance", "time_since_assist", "delivery_distance"]:
        df[c] = df[c].fillna(-1)
    # a handful of restarts lasting minutes are recycled possessions; cap them
    df["time_since_restart"] = df["time_since_restart"].clip(upper=30)
    df["events_since_restart"] = df["events_since_restart"].clip(upper=30)
    return df


def sp_logit():
    pre = ColumnTransformer([
        ("spl", SplineTransformer(n_knots=4, degree=3), ["distance", "goal_mouth_angle"]),
        ("num", StandardScaler(), SP_NUM + ["has_assist", "has_delivery"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), SP_CAT),
    ])
    return make_pipeline(pre, LogisticRegression(C=0.1, max_iter=4000))


class SPGBM:
    params = dict(
        objective="binary", learning_rate=0.03, n_estimators=250, num_leaves=6,
        min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_lambda=10.0, random_state=SEED, verbose=-1,
    )

    def _x(self, df):
        X = df[SP_NUM + ["has_assist", "has_delivery"] + SP_CAT].copy()
        for c in SP_CAT:
            X[c] = pd.Categorical(X[c], categories=self.cats_[c])
        return X

    def fit(self, df, y):
        self.cats_ = {c: sorted(df[c].unique()) for c in SP_CAT}
        self.model_ = lgb.LGBMClassifier(**self.params).fit(self._x(df), y)
        return self

    def predict_proba(self, df):
        return self.model_.predict_proba(self._x(df))


MODELS = {"sp_location": location_model, "sp_full_logit": sp_logit, "sp_full_gbm": SPGBM}


def main():
    os.makedirs(SP_DIR, exist_ok=True)
    df = load()
    is_pen = df["restart_type"] == "PENALTY"
    train = df[~is_pen].reset_index(drop=True)
    y = train["goal"].values

    oof = {name: np.zeros(len(train)) for name in MODELS}
    for tr, te in GroupKFold(n_splits=5).split(train, y, train["match_id"]):
        for name, make in MODELS.items():
            oof[name][te] = make().fit(train.iloc[tr], y[tr]).predict_proba(train.iloc[te])[:, 1]

    scores = {name: metrics(y, p) for name, p in oof.items()}
    scores["baseline_mean"] = metrics(y, np.full(len(y), y.mean()))
    general_path = os.path.join(OUT_DIR, "shots_with_xg.csv")
    if os.path.exists(general_path):
        general = pd.read_csv(general_path).set_index("event_id")["xg"]
        scores["general_model"] = metrics(y, general.loc[train.event_id].values)
    scores = pd.DataFrame(scores).T
    print(scores.round(4).to_string())
    scores.to_csv(os.path.join(SP_DIR, "model_scores.csv"))

    best = scores.drop(index="baseline_mean")["log_loss"].idxmin()
    print(f"\nBest model on set-piece shots by log loss: {best}")

    final = {name: make().fit(train, y) for name, make in MODELS.items()}
    joblib.dump({"models": final, "best": best, "penalty_xg": PENALTY_XG, "rare_min": RARE_MIN},
                os.path.join(SP_DIR, "setpiece_xg_models.joblib"))

    for name, p in oof.items():
        train[f"xg_{name}"] = p
    pens = df[is_pen].copy()
    for name in MODELS:
        pens[f"xg_{name}"] = PENALTY_XG
    shots = pd.concat([train, pens], ignore_index=True)
    if "general_model" in scores.index:
        shots["xg_general_model"] = general.reindex(shots.event_id).values
    shots["xg"] = shots[f"xg_{best}"]
    shots["team"] = shots["squad_id"].map(squad_names(pd.read_csv(os.path.join(OUT_DIR, "shots.csv"))))
    shots.sort_values(["league", "match_file", "period", "game_time_sec"]).to_csv(
        os.path.join(SP_DIR, "setpiece_shots_with_xg.csv"), index=False)

    plot_calibration(y, oof, os.path.join(SP_DIR, "calibration.png"))
    imp = pd.Series(final["sp_full_gbm"].model_.booster_.feature_importance("gain"),
                    index=final["sp_full_gbm"].model_.feature_name_)
    print("\nTop GBM features:\n", (imp / imp.sum()).sort_values(ascending=False).head(10).round(3).to_string())

    by_type = shots.groupby("restart_type").agg(shots=("goal", "size"), goals=("goal", "sum"), xg=("xg", "sum"),
                                               xg_per_shot=("xg", "mean")).round(3)
    by_type.to_csv(os.path.join(SP_DIR, "xg_by_restart_type.csv"))
    print("\nAll leagues by restart type:\n", by_type.to_string())

    detail = (shots[shots.restart_type.isin(["CORNER", "FREE_KICK", "THROW_IN"])]
              .assign(contact=lambda d: np.where(d.first_contact == 1, "first contact", "later"),
                      swing=lambda d: d.delivery_swing.map({1: "side A", -1: "side B"}).fillna("n/a"))
              .groupby(["restart_type", "contact", "swing"])
              .agg(shots=("goal", "size"), goals=("goal", "sum"), xg=("xg", "sum"), xg_per_shot=("xg", "mean"))
              .round(3))
    detail.to_csv(os.path.join(SP_DIR, "xg_by_delivery.csv"))
    print("\nBy delivery (swing: taker foot vs. pitch side, A/B = in- vs out-swing):\n", detail.to_string())

    ere = shots[shots.league.str.startswith("Eredivisie")]
    teams = (ere.groupby(["squad_id", "team"])
             .agg(matches=("match_id", "nunique"), shots=("goal", "size"), goals=("goal", "sum"), xg=("xg", "sum"),
                  corner_xg=("xg", lambda s: s[ere.loc[s.index, "restart_type"] == "CORNER"].sum()),
                  free_kick_xg=("xg", lambda s: s[ere.loc[s.index, "restart_type"].isin(["FREE_KICK", "DIRECT_FREE_KICK"])].sum()),
                  throw_in_xg=("xg", lambda s: s[ere.loc[s.index, "restart_type"] == "THROW_IN"].sum()),
                  penalty_xg=("xg", lambda s: s[ere.loc[s.index, "restart_type"] == "PENALTY"].sum()))
             .assign(xg_per_match=lambda d: d.xg / d.matches, goals_minus_xg=lambda d: d.goals - d.xg)
             .sort_values("xg", ascending=False).round(3))
    teams.to_csv(os.path.join(SP_DIR, "eredivisie_teams_setpiece_xg.csv"))
    print("\nEredivisie set-piece xG (for):\n", teams.to_string())

    players = (ere.groupby(["player_id", "player_name"])
               .agg(shots=("goal", "size"), goals=("goal", "sum"), xg=("xg", "sum"))
               .assign(goals_minus_xg=lambda d: d.goals - d.xg)
               .sort_values("xg", ascending=False).round(3))
    players.to_csv(os.path.join(SP_DIR, "eredivisie_players_setpiece_xg.csv"))
    print("\nEredivisie players top 10:\n", players.head(10).to_string())


if __name__ == "__main__":
    main()
