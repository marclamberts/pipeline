"""Classify goal-kick routines and rank them by how often they lead to shots / xG.

Run analysis/goalkick_extract.py first.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import cross_val_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "output")
MIN_N = 30          # minimum goal kicks for a routine to be ranked
PRIOR_N = 60        # strength of the league-average prior used for shrinkage


# ---------------------------------------------------------------- xG model
def xg_features(s):
    X = pd.DataFrame(index=s.index)
    X["log_dist"] = np.log(s.dist.clip(lower=1))
    X["angle"] = s.angle
    X["header"] = (s.body == "HEAD").astype(int) | (s.action == "HEADER").astype(int)
    for a in ["DIRECT_FREE_KICK", "ONE_VS_ONE_AGAINST_GK", "OPEN_GOAL_SHOT", "LONG_RANGE_SHOT", "CLOSE_RANGE_SHOT"]:
        X[a.lower()] = (s.action == a).astype(int)
    X["transition"] = (s.phase == "ATTACKING_TRANSITION").astype(int)
    X["pressure"] = s.pressure / 100
    return X


def fit_xg(shots):
    pen = shots.action == "PENALTY_KICK"
    s = shots[~pen]
    X, y = xg_features(s), s.goal
    model = LogisticRegression(C=1.0, max_iter=2000)
    p_cv = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:, 1]
    model.fit(X, y)
    shots = shots.copy()
    shots["xg"] = 0.76
    shots.loc[~pen, "xg"] = model.predict_proba(X)[:, 1]
    report = dict(n=len(s), goals=int(y.sum()), auc_cv=roc_auc_score(y, p_cv),
                  brier_cv=brier_score_loss(y, p_cv), xg_sum=shots.xg.sum(), goals_all=int(shots.goal.sum()))
    # sanity check against the provider's xG where available
    danger = glob.glob(os.path.join(ROOT, "*", "Danger", "*.csv"))
    if danger:
        dz = pd.concat(pd.read_csv(f) for f in danger)
        dz["match_id"] = dz.match_file.str.extract(r"_(\d+)\.json")[0]
        m = shots.assign(match_id=shots.match_id.astype(str))
        mm = m.merge(dz, left_on=["match_id", "squad", "x", "y"], right_on=["match_id", "squad_id", "x", "y"],
                     suffixes=("", "_impect"))
        if len(mm) > 5:
            report["corr_vs_impect_xg"] = float(np.corrcoef(mm.xg, mm.xg_impect)[0, 1])
            report["n_matched_impect"] = len(mm)
    return shots, report


# ---------------------------------------------------------------- routine labels
SIDE = {"L": "left", "C": "centre", "R": "right", None: "?"}


def setup_label(r):
    """Level 1: how the goal kick itself is played."""
    taker = "GK" if r.taker == "GK" else "CB"
    z = r.gk_zone
    if z == "long":
        return f"{taker} long → opp half ({SIDE.get(r.gk_side, '?')})"
    if z == "mid":
        return f"{taker} → own half, outside 3rd ({r.rec_pos or '?'})"
    if z in ("short_box",):
        rec = r.rec_pos if r.rec_pos in ("GK", "CB") else "other"
        return f"{taker} short in box → {rec}"
    if z in ("short_edge", "short_wide"):
        rec = r.rec_pos if r.rec_pos in ("CB", "LB", "RB", "DM", "GK") else "other"
        rec = "FB" if rec in ("LB", "RB") else rec
        return f"{taker} short outside box → {rec}"
    return "unknown"


def plan_label(r):
    """Level 2: what follows the short goal kick."""
    if r.gk_zone == "long":
        return "direct"
    if pd.notna(r.launch_k) and r.launch_k > 0:
        who = "GK" if r.launch_by == "GK" else ("CB" if r.launch_by == "CB" else "other")
        return f"then long by {who} ({SIDE.get(r.launch_side if isinstance(r.launch_side, str) else None)})"
    route = r.exit_route if isinstance(r.exit_route, str) else None
    if route is None:
        return "build, lost in box"
    return f"build via {SIDE[route]}"


def rate_table(df, by, min_n=MIN_N):
    base_shot, base_xg = df.shot.mean(), df.xg_chain.mean()
    t = df.groupby(by).agg(
        n=("shot", "size"), shots=("n_shots", "sum"), shot_rate=("shot", "mean"),
        xg_per_gk=("xg_chain", "mean"), goals=("goal", "sum"),
        reach_final3=("reach_final3", "mean"), lost_own3=("lost_own3", "mean"),
        lift_within_team=("shot_lift", "mean"),
        xg_against=("xg_against", "mean"), shot_against=("shot_against", "mean"), net_xg=("net_xg", "mean"),
        med_t_shot=("t_first_shot", "median"), teams=("team", "nunique"),
    ).reset_index()
    t = t[t.n >= min_n].copy()
    # empirical-Bayes shrinkage towards the overall average
    t["shot_rate_adj"] = (t.shot_rate * t.n + base_shot * PRIOR_N) / (t.n + PRIOR_N)
    t["xg_per_gk_adj"] = (t.xg_per_gk * t.n + base_xg * PRIOR_N) / (t.n + PRIOR_N)
    base_net = df.net_xg.mean()
    t["net_xg_adj"] = (t.net_xg * t.n + base_net * PRIOR_N) / (t.n + PRIOR_N)
    # Wilson 90% CI on shot rate
    z = 1.645
    p, n = t.shot_rate, t.n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    t["shot_ci_lo"], t["shot_ci_hi"] = c - h, c + h
    t["share"] = t.n / len(df)
    return t.sort_values("xg_per_gk_adj", ascending=False)


def main():
    shots = pd.read_csv(os.path.join(OUT, "shots.csv"))
    gks = pd.read_csv(os.path.join(OUT, "goalkicks.csv"))
    teams = pd.read_csv(os.path.join(OUT, "teams.csv")).set_index("squad").team_name

    shots, rep = fit_xg(shots)
    print("xG model:", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in rep.items()}))
    shots.to_csv(os.path.join(OUT, "shots_xg.csv"), index=False)
    xg_lookup = shots.set_index(["match_id", "idx"])
    goal_lookup = xg_lookup.goal

    def chain_xg(r, col="shot_idx"):
        ids = json.loads(getattr(r, col))
        xs = [xg_lookup.xg.get((r.match_id, i), 0) for i in ids]
        # probability of at least one goal (shots in the same move are not independent)
        return 1 - np.prod([1 - x for x in xs]) if xs else 0.0

    gks = gks[gks.gk_zone != "unknown"].copy()
    gks["xg_chain"] = gks.apply(chain_xg, axis=1)
    gks["xg_against"] = gks.apply(lambda r: chain_xg(r, "opp_shot_idx"), axis=1)
    gks["net_xg"] = gks.xg_chain - gks.xg_against
    gks["shot_against"] = (gks.opp_shot_idx != "[]").astype(int)
    gks["goal"] = gks.apply(lambda r: int(any(goal_lookup.get((r.match_id, i), 0) for i in json.loads(r.shot_idx))), axis=1)
    gks["shot"] = (gks.n_shots > 0).astype(int)
    gks["reach_final3"] = gks.t_final.notna().astype(int)
    gks["lost_own3"] = gks.lost_own_third.fillna(0).astype(int)
    gks["team_name"] = gks.team.map(teams)
    gks["opp_name"] = gks.opp.map(teams)
    gks["setup"] = gks.apply(setup_label, axis=1)
    gks["plan"] = gks.apply(plan_label, axis=1)
    gks["routine"] = gks.setup + " | " + gks.plan
    # within-team lift: routine outcome minus that team's average goal-kick shot rate
    gks["shot_lift"] = gks.shot - gks.groupby("team").shot.transform("mean")

    gks.to_csv(os.path.join(OUT, "goalkicks_labelled.csv"), index=False)

    overall = dict(n=len(gks), shot_rate=gks.shot.mean(), xg_per_gk=gks.xg_chain.mean(), goals=int(gks.goal.sum()),
                   shot_rate_same_possession=gks.shot_in_seq.mean())
    print("overall:", overall)
    print(gks.groupby("league")[["shot", "xg_chain"]].mean().round(3))

    tabs = {
        "by_setup": rate_table(gks, "setup"),
        "by_plan": rate_table(gks, "plan"),
        "by_routine": rate_table(gks, "routine"),
        "by_team": rate_table(gks, "team_name", min_n=15),
        "by_team_routine": rate_table(gks, ["team_name", "routine"], min_n=8),
        "by_opp_formation": rate_table(gks, ["setup", "opp_formation"], min_n=25),
    }
    gks["strategy"] = np.where(gks.gk_zone == "long", "1 Direct long",
                       np.where(gks.plan.str.startswith("then long"), "2 Short, then long", "3 Short, play out"))
    tabs["by_strategy"] = rate_table(gks, "strategy")
    tabs["by_league_strategy"] = rate_table(gks, ["league", "strategy"])
    # pass-path patterns (positions of the first 3 passers)
    gks["path3"] = gks.path_pos.str.split("-").str[:3].str.join("-") + " | " + gks.plan
    tabs["by_path"] = rate_table(gks, "path3")
    for k, t in tabs.items():
        t.to_csv(os.path.join(OUT, f"{k}.csv"), index=False)

    cols = ["n", "share", "shot_rate", "shot_ci_lo", "shot_ci_hi", "xg_per_gk", "xg_per_gk_adj", "goals",
            "reach_final3", "lost_own3", "shot_against", "xg_against", "net_xg", "lift_within_team", "med_t_shot"]
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
    for k in ["by_strategy", "by_league_strategy", "by_setup", "by_plan", "by_routine", "by_path"]:
        t = tabs[k]
        print(f"\n=== {k}")
        t = t.sort_values("net_xg_adj", ascending=False)
        print(t[[c for c in t.columns if c not in cols and c not in ("shots", "shot_rate_adj", "teams")] + cols].round(3).to_string(index=False))
    print("\n=== by_team (top 15 by adj xG/GK)")
    print(tabs["by_team"].head(15)[["team_name"] + cols].round(3).to_string(index=False))
    print("\n=== best team-routine combos (n>=8)")
    t = tabs["by_team_routine"].sort_values("xg_per_gk", ascending=False)
    print(t.head(20)[["team_name", "routine"] + cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
