"""
Transition Anatomy Model - 3. Liga 2026-27 (Impect event data)

A descriptive, rule-based model of attacking transitions. It deliberately uses
NO expected-threat (xT / pxT) values and NO predictive / fitted probabilities.
Every number is something that was observed on the pitch:

  1. Trigger      - how the ball was won (interception, tackle, loose ball, ...)
  2. Regain zone  - where it was won (thirds x lanes)
  3. Context      - pressure on the regainer, losing team's players goal-side
                    ("rest defence") at the moment of the turnover
  4. Tempo        - seconds to final third / box, vertical speed (m/s)
  5. Directness   - forward metres gained / total metres the ball travelled
  6. Packing      - opponents bypassed (Impect `opponents` = opponents goal-side)
  7. Reach ladder - furthest stage reached: own half < middle < final third
                    < box < shot < goal (observed frequencies, no modelling)
  8. Ending       - how it ended: settled possession, set piece won, out of
                    play, lost, tactical foul suffered, shot
  9. Archetype    - transparent rules on the above (no clustering, no fitting)

Defensive mirror (team as the side that just lost the ball):
  - counter-press kill rate: opponent transitions ended by a regain within 5 s
  - tactical-foul rate: opponent transitions stopped by a foul
  - rest-defence count: own players goal-side when the ball is lost

Usage:
    python3 transitions/transition_model.py ["3. Liga 2026-2027"]
Outputs land in transitions/output/.
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEAGUE_DIR = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "3. Liga 2026-2027")
OUT_DIR = os.path.join(ROOT, "transitions", "output")

# Pitch geometry (adjCoordinates: team always attacks towards +x, 105 x 68)
HALF_X = 52.5
FINAL_THIRD_X = HALF_X - 35.0          # 17.5
BOX_X = HALF_X - 16.5                  # 36.0
BOX_HALF_Y = 20.16
COUNTERPRESS_WINDOW = 5.0              # seconds
POSSESSION_FOLLOW_UP = 20.0            # seconds after transition to look for a shot

REGAIN_TYPES = {"INTERCEPTION", "LOOSE_BALL_REGAIN", "GROUND_DUEL", "GK_CATCH",
                "GK_SAVE", "CLEARANCE", "BLOCK", "AERIAL_DUEL"}

NAME_FIX = {
    "Preu_en_M_nster": "Preußen Münster",
    "1_FC_Saarbr_cken": "1. FC Saarbrücken",
    "W_rzburger_Kickers": "Würzburger Kickers",
    "FC_Viktoria_K_ln": "FC Viktoria Köln",
    "SC_Fortuna_K_ln": "SC Fortuna Köln",
    "SG_Sonnenhof_Gro_aspach": "SG Sonnenhof Großaspach",
    "Fortuna_D_sseldorf": "Fortuna Düsseldorf",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def clean_name(raw):
    return NAME_FIX.get(raw, raw.replace("_", " "))


def parse_filename(path):
    m = re.match(r"(\d{4}-\d{2}-\d{2})_(.+)_vs_(.+)_(\d+)\.json$", os.path.basename(path))
    date, home, away, match_id = m.groups()
    return date, clean_name(home), clean_name(away), int(match_id)


def t_abs(e):
    """Continuous match clock in seconds (period 2 is stored from 10000 s)."""
    t = e["gameTime"]["gameTimeInSec"]
    return t - 10000 + 45 * 60 if e["periodId"] == 2 else t


def xy(e):
    s = e.get("start") or {}
    c = s.get("adjCoordinates") or {}
    if c.get("x") is None or c.get("y") is None:
        return None
    return float(c["x"]), float(c["y"])


def zone(x, y):
    third = "DEF" if x < -HALF_X / 3 else ("MID" if x < HALF_X / 3 else "ATT")
    lane = "L" if y > 68 / 6 else ("R" if y < -68 / 6 else "C")
    return third, lane


def in_box(p):
    return p[0] >= BOX_X and abs(p[1]) <= BOX_HALF_Y


def load_matches():
    files = sorted(f for f in glob.glob(os.path.join(LEAGUE_DIR, "*.json"))
                   if os.path.getsize(f) > 10)
    matches = []
    for f in files:
        with open(f) as fh:
            ev = json.load(fh)
        if ev:
            matches.append((f, ev))
    return matches


def squad_names(matches):
    """Map squadId -> team name by intersecting ids across fixtures."""
    cand = defaultdict(lambda: None)
    for f, ev in matches:
        _, home, away, _ = parse_filename(f)
        ids = {e["squadId"] for e in ev if e["squadId"] is not None}
        for name in (home, away):
            cand[name] = ids if cand[name] is None else cand[name] & ids
    out = {}
    for name, ids in cand.items():
        if ids and len(ids) == 1:
            out[next(iter(ids))] = name
    return out


# --------------------------------------------------------------------------- #
# Transition extraction
# --------------------------------------------------------------------------- #
def find_runs(ev):
    """Maximal runs of consecutive ATTACKING_TRANSITION events for one squad."""
    runs, cur = [], []
    for e in ev:
        if e["phase"] == "ATTACKING_TRANSITION" and e["currentAttackingSquadId"] is not None:
            if cur and (cur[-1]["currentAttackingSquadId"] != e["currentAttackingSquadId"]
                        or cur[-1]["periodId"] != e["periodId"]):
                runs.append(cur)
                cur = []
            cur.append(e)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def describe_run(run, ev, match_meta, names):
    team = run[0]["currentAttackingSquadId"]
    own = [e for e in run if e["squadId"] == team]
    if not own:
        return None

    # Walk back over the team's own SECOND_BALL events that led into the run:
    # the regain itself often happens in the contested second-ball phase.
    first_idx = own[0]["index"]
    k = first_idx - 1
    lead_in = []
    while k >= 0 and ev[k]["squadId"] == team and ev[k]["phase"] == "SECOND_BALL" \
            and ev[k]["periodId"] == own[0]["periodId"]:
        lead_in.insert(0, ev[k])
        k -= 1
    regain = lead_in[0] if lead_in else own[0]
    prev = ev[regain["index"] - 1] if regain["index"] > 0 else None
    chain = lead_in + own

    # Trigger
    if lead_in:
        trigger = "SECOND_BALL_WIN"
    elif regain["actionType"] in REGAIN_TYPES:
        trigger = regain["actionType"]
    elif prev is not None and prev["actionType"] == "PASS" and prev["result"] == "FAIL":
        trigger = "OPP_MISPLACED_PASS"
    elif prev is not None and prev["actionType"] == "SHOT":
        trigger = "AFTER_OPP_SHOT"
    else:
        trigger = "OTHER"

    pts = [(e, xy(e)) for e in chain]
    pts = [(e, p) for e, p in pts if p is not None]
    if not pts:
        return None
    t0 = t_abs(regain)
    x0, y0 = pts[0][1]
    third0, lane0 = zone(x0, y0)

    # Path, progress, tempo
    xs = np.array([p[0] for _, p in pts])
    path = sum(np.hypot(pts[i + 1][1][0] - pts[i][1][0], pts[i + 1][1][1] - pts[i][1][1])
               for i in range(len(pts) - 1))
    i_max = int(xs.argmax())
    x_max = float(xs[i_max])
    progress = max(0.0, x_max - x0)
    t_max = t_abs(pts[i_max][0]) - t0
    t_end = t_abs(chain[-1]) - t0
    directness = progress / path if path > 1 else np.nan
    speed = progress / t_max if t_max > 0.5 else np.nan

    t_ft = next((t_abs(e) - t0 for e, p in pts if p[0] >= FINAL_THIRD_X and x0 < FINAL_THIRD_X), np.nan)
    t_box = next((t_abs(e) - t0 for e, p in pts if in_box(p)), np.nan)

    # Packing: opponents goal-side at regain vs. fewest reached
    opp_series = [e["opponents"] for e in chain if e.get("opponents") is not None]
    opp_start = opp_series[0] if opp_series else np.nan
    bypassed = (opp_series[0] - min(opp_series)) if opp_series else np.nan

    passes = [e for e in own if e["actionType"] == "PASS"]
    lanes = {zone(*p)[1] for _, p in pts}
    carries = [e for e in own if e.get("dribble")]
    prog_carries = [e for e in carries if e["dribble"]["type"] in ("PROGRESSIVE_CARRY", "ONE_VS_ONE")]
    first_pass = passes[0] if passes else None
    long_release = bool(first_pass and first_pass["pass"] and (first_pass["pass"].get("distance") or 0) >= 30)
    players = {e["player"]["id"] for e in own if e.get("player")}

    shot_in_run = any(e["actionType"] == "SHOT" and e["squadId"] == team for e in run)
    goal_in_run = any(e["actionType"] == "GOAL" and e["squadId"] == team for e in run)

    # What happened right after the run?
    nxt = ev[run[-1]["index"] + 1] if run[-1]["index"] + 1 < len(ev) else None
    last_own = own[-1]
    opp = None
    if shot_in_run:
        ending = "SHOT"
    elif nxt is None:
        ending = "END_OF_PERIOD"
    elif nxt["currentAttackingSquadId"] == team and nxt["phase"] == "IN_POSSESSION":
        ending = "SETTLED"
    elif nxt["phase"] == "SET_PIECE" and nxt["squadId"] == team:
        foul = any(e["actionType"] == "FOUL" and e["squadId"] != team for e in run[-3:])
        ending = "FOULED" if foul or nxt["actionType"] == "FREE_KICK" else "WON_SET_PIECE"
    elif nxt["phase"] == "SET_PIECE" or last_own["actionType"] in ("OUT", "OFFSIDE") \
            or run[-1]["actionType"] in ("OUT", "OFFSIDE"):
        ending = "OUT_OF_PLAY"
    elif nxt["phase"] == "SECOND_BALL" and nxt["squadId"] == team:
        ending = "CONTESTED_KEPT"
    else:
        ending = "LOST"
    if nxt is not None and nxt["squadId"] not in (team, None):
        opp = nxt["squadId"]

    # Shot later in the same possession (continuation after settling)
    shot_follow = shot_in_run
    goal_follow = goal_in_run
    if not shot_in_run and ending in ("SETTLED", "CONTESTED_KEPT", "FOULED", "WON_SET_PIECE"):
        j = run[-1]["index"] + 1
        while j < len(ev) and t_abs(ev[j]) - t0 <= t_end + POSSESSION_FOLLOW_UP \
                and ev[j]["currentAttackingSquadId"] == team:
            if ev[j]["actionType"] == "SHOT" and ev[j]["squadId"] == team:
                shot_follow = True
            if ev[j]["actionType"] == "GOAL" and ev[j]["squadId"] == team:
                goal_follow = True
            j += 1

    # Reach ladder (observed stage, ordinal)
    if goal_in_run:
        reach = 5
    elif shot_in_run:
        reach = 4
    elif not np.isnan(t_box):
        reach = 3
    elif x_max >= FINAL_THIRD_X:
        reach = 2
    elif x_max >= 0:
        reach = 1
    else:
        reach = 0

    killed_fast = ending == "LOST" and t_end <= COUNTERPRESS_WINDOW

    # Archetype - transparent rules, evaluated in order
    if third0 == "ATT":
        archetype = "High regain strike" if reach >= 3 else "High regain recycle"
    elif killed_fast:
        archetype = "Immediate loss"
    elif long_release and progress >= 20:
        archetype = "Long-ball release"
    elif reach >= 3 and (speed or 0) >= 4.0:
        archetype = "Direct counter" if len(passes) <= 3 else "Combined counter"
    elif reach >= 3:
        archetype = "Slow build to box"
    elif ending == "SETTLED" and progress < 15:
        archetype = "Secure & reset"
    elif progress >= 20:
        archetype = "Progress then stalled"
    else:
        archetype = "Stalled early"

    date, home, away, match_id = match_meta
    defender = next((s for s in {home, away} if s != names.get(team)), None)
    return {
        "match_id": match_id, "date": date, "home": home, "away": away,
        "team_id": team, "team": names.get(team, str(team)), "opponent": defender,
        "period": regain["periodId"], "minute": round(t0 / 60, 1),
        "regain_player": (regain.get("player") or {}).get("name"),
        "regain_player_id": (regain.get("player") or {}).get("id"),
        "regain_position": (regain.get("player") or {}).get("position"),
        "trigger": trigger,
        "regain_x": round(x0, 1), "regain_y": round(y0, 1),
        "regain_third": third0, "regain_lane": lane0,
        "pressure_at_regain": regain.get("pressure"),
        "opp_goal_side_at_regain": opp_start,
        "n_actions": len(own), "n_passes": len(passes), "n_players": len(players),
        "n_prog_carries": len(prog_carries), "long_release": long_release,
        "lanes_used": len(lanes),
        "duration_s": round(t_end, 2),
        "progress_m": round(progress, 1), "path_m": round(path, 1),
        "directness": round(directness, 3) if not np.isnan(directness) else np.nan,
        "vertical_speed_ms": round(speed, 2) if not np.isnan(speed) else np.nan,
        "t_to_final_third_s": round(t_ft, 2) if not np.isnan(t_ft) else np.nan,
        "t_to_box_s": round(t_box, 2) if not np.isnan(t_box) else np.nan,
        "opponents_bypassed": bypassed,
        "reach": reach,
        "reached_final_third": reach >= 2,
        "reached_box": reach >= 3,
        "shot": shot_in_run, "goal": goal_in_run,
        "shot_in_possession": shot_follow, "goal_in_possession": goal_follow,
        "ending": ending, "killed_within_5s": killed_fast,
        "archetype": archetype,
    }, chain


def player_credit(chain, team, store, tid):
    """Opponents bypassed and metres gained between consecutive own actions."""
    own = [e for e in chain if e["squadId"] == team and e.get("player")]
    for a, b in zip(own, own[1:]):
        pid = a["player"]["id"]
        rec = store[pid]
        rec["name"] = a["player"]["name"]
        rec["team_id"] = tid
        if a.get("opponents") is not None and b.get("opponents") is not None:
            rec["bypassed"] += max(0, a["opponents"] - b["opponents"])
        pa, pb = xy(a), xy(b)
        if pa and pb and a["actionType"] in ("PASS", "DRIBBLE"):
            rec["metres_gained"] += max(0.0, pb[0] - pa[0])
    for e in own:
        store[e["player"]["id"]]["actions"] += 1


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def team_profiles(df, games):
    g = df.groupby("team")
    att = pd.DataFrame({
        "games": games,
        "transitions_pg": g.size() / games,
        "pct_high_regain": g["regain_third"].apply(lambda s: (s == "ATT").mean()),
        "avg_regain_x": g["regain_x"].mean(),
        "med_duration_s": g["duration_s"].median(),
        "med_vertical_speed_ms": g["vertical_speed_ms"].median(),
        "med_directness": g["directness"].median(),
        "avg_opponents_bypassed": g["opponents_bypassed"].mean(),
        "pct_reach_final_third": g["reached_final_third"].mean(),
        "pct_reach_box": g["reached_box"].mean(),
        "pct_shot": g["shot"].mean(),
        "pct_shot_in_possession": g["shot_in_possession"].mean(),
        "goals_from_transitions": g["goal_in_possession"].sum(),
        "pct_settled": g["ending"].apply(lambda s: (s == "SETTLED").mean()),
        "pct_lost_within_5s": g["killed_within_5s"].mean(),
        "med_t_to_box_s": g["t_to_box_s"].median(),
    })
    d = df.groupby("opponent")
    games_d = df.groupby("opponent")["match_id"].nunique()
    dfn = pd.DataFrame({
        "opp_transitions_pg": d.size() / games_d,
        "opp_pct_reach_box": d["reached_box"].mean(),
        "opp_pct_shot_in_possession": d["shot_in_possession"].mean(),
        "counterpress_kill_rate": d["killed_within_5s"].mean(),
        "tactical_foul_rate": d["ending"].apply(lambda s: (s == "FOULED").mean()),
        # only turnovers the team suffered in its attacking half (regain_x < 0
        # from the regaining side) - that is where rest defence matters
        "rest_defence_goal_side": df[df["regain_x"] < 0].groupby("opponent")["opp_goal_side_at_regain"].mean(),
        "opp_avg_regain_x": d["regain_x"].mean(),
    })
    prof = att.join(dfn, how="left")
    prof.index.name = "team"
    return prof.sort_values("pct_reach_box", ascending=False)


def archetype_mix(df):
    mix = pd.crosstab(df["team"], df["archetype"], normalize="index")
    return mix[sorted(mix.columns)]


def outcome_ladder(df):
    """Empirical (not modelled) reach rates by regain zone and trigger."""
    by_zone = df.groupby(["regain_third", "regain_lane"]).agg(
        n=("reach", "size"),
        pct_final_third=("reached_final_third", "mean"),
        pct_box=("reached_box", "mean"),
        pct_shot_in_possession=("shot_in_possession", "mean"),
        med_duration_s=("duration_s", "median"),
    )
    by_trigger = df.groupby("trigger").agg(
        n=("reach", "size"),
        avg_regain_x=("regain_x", "mean"),
        pct_box=("reached_box", "mean"),
        pct_shot_in_possession=("shot_in_possession", "mean"),
        pct_lost_within_5s=("killed_within_5s", "mean"),
    ).sort_values("n", ascending=False)
    by_bypass = df.assign(bypass_band=pd.cut(df["opponents_bypassed"], [-1, 0, 2, 4, 6, 11],
                                             labels=["0", "1-2", "3-4", "5-6", "7+"])) \
        .groupby("bypass_band", observed=True).agg(
            n=("reach", "size"),
            pct_box=("reached_box", "mean"),
            pct_shot_in_possession=("shot_in_possession", "mean"))
    return by_zone, by_trigger, by_bypass


def fmt_pct(x):
    return f"{x * 100:.0f}%"


def write_report(df, prof, mix, ladder, players, n_matches):
    by_zone, by_trigger, by_bypass = ladder
    L = []
    L.append("# Transition Anatomy - 3. Liga 2026-27\n")
    L.append(f"Matches with data: **{n_matches}** · attacking transitions: **{len(df)}** · "
             f"date range {df['date'].min()} to {df['date'].max()}\n")
    L.append("No xT/pxT values and no fitted probabilities are used. "
             "Every figure is an observed count, time, distance or share.\n")

    L.append("## League baseline\n")
    L.append(f"- Median transition: {df['duration_s'].median():.1f} s, "
             f"{df['n_actions'].median():.0f} actions, {df['progress_m'].median():.0f} m forward")
    L.append(f"- Reach final third {fmt_pct(df['reached_final_third'].mean())}, box "
             f"{fmt_pct(df['reached_box'].mean())}, shot in possession {fmt_pct(df['shot_in_possession'].mean())}")
    L.append(f"- Killed by the opponent within 5 s: {fmt_pct(df['killed_within_5s'].mean())}")
    L.append(f"- Endings: " + ", ".join(f"{k} {fmt_pct(v)}" for k, v in
                                        df['ending'].value_counts(normalize=True).items()) + "\n")

    L.append("## Archetypes (rule-based)\n")
    a = df.groupby("archetype").agg(n=("reach", "size"), pct_shot=("shot_in_possession", "mean"),
                                    med_dur=("duration_s", "median"),
                                    goals=("goal_in_possession", "sum")).sort_values("n", ascending=False)
    L.append("| Archetype | n | share | shot in possession | median s | goals |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for k, r in a.iterrows():
        L.append(f"| {k} | {int(r.n)} | {fmt_pct(r.n / len(df))} | {fmt_pct(r.pct_shot)} | {r.med_dur:.1f} | {int(r.goals)} |")
    L.append("")

    L.append("## Where regains happen and what they become (empirical ladder)\n")
    L.append("| Third | Lane | n | → final third | → box | → shot |")
    L.append("|---|---|---:|---:|---:|---:|")
    for (t, l), r in by_zone.iterrows():
        L.append(f"| {t} | {l} | {int(r.n)} | {fmt_pct(r.pct_final_third)} | {fmt_pct(r.pct_box)} | {fmt_pct(r.pct_shot_in_possession)} |")
    L.append("")

    L.append("## Trigger\n")
    L.append("| Trigger | n | avg regain x | → box | → shot | lost ≤5 s |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for k, r in by_trigger.iterrows():
        L.append(f"| {k} | {int(r.n)} | {r.avg_regain_x:.1f} | {fmt_pct(r.pct_box)} | {fmt_pct(r.pct_shot_in_possession)} | {fmt_pct(r.pct_lost_within_5s)} |")
    L.append("")

    L.append("## Opponents bypassed (packing) vs. outcome\n")
    L.append("| Bypassed | n | → box | → shot |")
    L.append("|---|---:|---:|---:|")
    for k, r in by_bypass.iterrows():
        L.append(f"| {k} | {int(r.n)} | {fmt_pct(r.pct_box)} | {fmt_pct(r.pct_shot_in_possession)} |")
    L.append("")

    L.append("## Team profiles - attacking transitions\n")
    L.append("| Team | G | per game | high regain | regain x | speed m/s | directness | bypassed | → box | → shot | lost ≤5 s |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, r in prof.iterrows():
        L.append(f"| {k} | {int(r.games)} | {r.transitions_pg:.1f} | {fmt_pct(r.pct_high_regain)} | {r.avg_regain_x:.1f} | "
                 f"{r.med_vertical_speed_ms:.1f} | {r.med_directness:.2f} | {r.avg_opponents_bypassed:.2f} | "
                 f"{fmt_pct(r.pct_reach_box)} | {fmt_pct(r.pct_shot_in_possession)} | {fmt_pct(r.pct_lost_within_5s)} |")
    L.append("")

    L.append("## Team profiles - defending transitions (after losing the ball)\n")
    L.append("| Team | opp transitions / g | counter-press kill | tactical foul | rest defence (goal-side) | opp → box | opp → shot |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for k, r in prof.sort_values("counterpress_kill_rate", ascending=False).iterrows():
        L.append(f"| {k} | {r.opp_transitions_pg:.1f} | {fmt_pct(r.counterpress_kill_rate)} | {fmt_pct(r.tactical_foul_rate)} | "
                 f"{r.rest_defence_goal_side:.2f} | {fmt_pct(r.opp_pct_reach_box)} | {fmt_pct(r.opp_pct_shot_in_possession)} |")
    L.append("")

    L.append("## Players - transition carriers (opponents bypassed with own actions)\n")
    L.append("| Player | Team | actions | opponents bypassed | metres gained | regains started | → box from their regains |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for _, r in players.head(20).iterrows():
        L.append(f"| {r['name']} | {r['team']} | {int(r.actions)} | {int(r.bypassed)} | {r.metres_gained:.0f} | "
                 f"{int(r.regains)} | {int(r.regains_to_box)} |")
    L.append("")
    L.append("Method and definitions: see `transitions/README.md`.")
    return "\n".join(L) + "\n"


def main():
    matches = load_matches()
    names = squad_names(matches)
    rows, pstore = [], defaultdict(lambda: defaultdict(float))
    for f, ev in matches:
        meta = parse_filename(f)
        for run in find_runs(ev):
            res = describe_run(run, ev, meta, names)
            if res is None:
                continue
            row, chain = res
            rows.append(row)
            player_credit(chain, row["team_id"], pstore, row["team_id"])
    df = pd.DataFrame(rows)
    games = df.groupby("team")["match_id"].nunique()

    prof = team_profiles(df, games)
    mix = archetype_mix(df)
    ladder = outcome_ladder(df)

    players = pd.DataFrame.from_dict({k: dict(v) for k, v in pstore.items()}, orient="index")
    players.index.name = "player_id"
    players["team"] = players["team_id"].map(names)
    reg = df.groupby("regain_player_id").agg(regains=("reach", "size"), regains_to_box=("reached_box", "sum"))
    players = players.join(reg, how="left").fillna({"regains": 0, "regains_to_box": 0})
    players = players.sort_values("bypassed", ascending=False)

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "transitions.csv"), index=False)
    prof.round(3).to_csv(os.path.join(OUT_DIR, "team_profiles.csv"))
    mix.round(3).to_csv(os.path.join(OUT_DIR, "team_archetype_mix.csv"))
    ladder[0].round(3).to_csv(os.path.join(OUT_DIR, "ladder_by_zone.csv"))
    ladder[1].round(3).to_csv(os.path.join(OUT_DIR, "ladder_by_trigger.csv"))
    ladder[2].round(3).to_csv(os.path.join(OUT_DIR, "ladder_by_bypassed.csv"))
    players.round(1).to_csv(os.path.join(OUT_DIR, "player_transitions.csv"))
    with open(os.path.join(OUT_DIR, "REPORT.md"), "w") as fh:
        fh.write(write_report(df, prof, mix, ladder, players, len(matches)))
    print(f"{len(matches)} matches, {len(df)} transitions -> {OUT_DIR}")


if __name__ == "__main__":
    main()
