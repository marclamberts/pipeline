"""Match report: numbers, tables and charts for one match.

    python -m analysis.match_report Bayern Stuttgart [--date 2026-08-28]
    python -m analysis.match_report --match-id 268382

Writes reports/match/<match>/ with report.md (data sections), stats.json and PNG charts.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from analysis.footy import pitch
from analysis.footy.cache import bundle
from analysis.footy.data import REPO, find_match, score
from analysis.footy.metrics import goals_table, md_table, player_table, team_stats

STAT_ROWS = [
    ("Goals", "goals"), ("xG", "xg"), ("Non-penalty xG", "np_xg"), ("Shots", "shots"),
    ("Shots on target", "shots_on_target"), ("Big chances (xG ≥ 0.3)", "big_chances"), ("xG per shot", "xg_per_shot"),
    ("Set-piece-phase xG (all restarts)", "set_piece_xg"), ("Possession % (pass share)", "possession_pct"), ("Passes", "passes"),
    ("Pass accuracy %", "pass_accuracy_pct"), ("Long passes %", "long_pass_pct"),
    ("Progressive passes", "progressive_passes"), ("Final-third entries", "final_third_entries"),
    ("Box entries", "box_entries"), ("Crosses", "crosses"), ("Field tilt %", "field_tilt_pct"),
    ("PPDA (lower = more pressing)", "ppda"), ("Regains", "regains"), ("High regains (≤40 m from goal)", "high_regains"),
    ("Unsuccessful passes + dribbles", "losses"), ("…of which ground passes/dribbles", "cheap_losses"),
    ("Ground losses in own half", "own_half_losses"),
    ("Avg pressure on opp. build-up", "pressure_on_opp_buildup"), ("Formation", "formation"),
    ("Yellow / red cards", None),
]


def build(match) -> dict:
    df, shots, sp = bundle(match)
    teams = (match.home, match.away)
    stats = {t: team_stats(df, t, shots) for t in teams}
    goals = goals_table(df)
    players = {t: player_table(df, t, shots) for t in teams}

    out = REPO / "reports" / "match" / match.slug
    out.mkdir(parents=True, exist_ok=True)
    sc = score(df)
    title = f"{match.home} {sc[match.home]}–{sc[match.away]} {match.away}"
    pitch.shot_map(shots, teams, out / "shot_map.png", f"Shot map — {title}")
    pitch.xg_race(shots, goals, teams, out / "xg_race.png", f"xG race — {title}")
    pitch.lane_bars(stats[teams[0]]["entries_by_lane"], out / "entries_by_lane.png",
                    "Final-third entries by lane", against=stats[teams[1]]["entries_by_lane"], labels=teams,
                    subtitle="each team's own left / right, in the direction it attacks")

    # Halves split, useful for "who controlled when".
    halves = {}
    for p in (1, 2):
        s = shots[shots.period == p]
        halves[p] = {t: {"xg": round(float(s[s.team == t].xg.sum()), 2), "shots": int((s.team == t).sum())} for t in teams}

    sp_rows = []
    for t in teams:
        mine = sp[(sp.team == t) & sp.type.isin(["CORNER", "FREE_KICK", "PENALTY"])]
        mine = mine[(mine.type != "FREE_KICK") | mine.attacking_third.fillna(False)]
        sp_rows.append({"team": t, "corners": int((mine.type == "CORNER").sum()),
                        "att. free kicks": int((mine.type == "FREE_KICK").sum()),
                        "penalties": int((mine.type == "PENALTY").sum()),
                        "set-piece shots": int(mine.shots.sum()), "set-piece xG": round(float(mine.xg.sum()), 2),
                        "goals": int(mine.goals.sum())})

    data = {
        "match": {"id": match.match_id, "league": match.league, "date": match.date,
                  "home": match.home, "away": match.away, "score": sc},
        "team_stats": stats, "goals": goals, "halves": halves, "set_pieces": sp_rows,
        "top_players": {t: players[t].head(8).to_dict("records") for t in teams},
        "shots": shots.drop(columns=["idx", "squad_id", "player_id"]).round(3).to_dict("records"),
    }
    (out / "stats.json").write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str))

    # ---- markdown (data sections; the agent adds the narrative on top)
    h, a = teams
    lines = [f"# {title}", "", f"_{match.league} · {match.date} · match id {match.match_id}_", "",
             "<!-- NARRATIVE: summary goes here -->", "", "## Goals", ""]
    if goals:
        lines += [md_table(pd.DataFrame(goals)[["clock", "team", "scorer", "own_goal", "type", "phase", "xg", "assist"]])]
    else:
        lines += ["_No goals._"]
    lines += ["", "## Team stats", "", f"| | {h} | {a} |", "|---|---|---|"]
    for label, key in STAT_ROWS:
        if key is None:
            vals = [f"{stats[t]['yellow_cards']} / {stats[t]['red_cards']}" for t in teams]
        else:
            vals = ["" if stats[t][key] is None else stats[t][key] for t in teams]
        lines.append(f"| {label} | {vals[0]} | {vals[1]} |")
    lines += ["", "### By half", "", f"| | {h} xG (shots) | {a} xG (shots) |", "|---|---|---|"]
    for p in (1, 2):
        lines.append(f"| {'1st' if p == 1 else '2nd'} half | {halves[p][h]['xg']} ({halves[p][h]['shots']}) | "
                     f"{halves[p][a]['xg']} ({halves[p][a]['shots']}) |")
    lines += ["", "### xG by phase", "", f"| phase | {h} | {a} |", "|---|---|---|"]
    phases = sorted(set(stats[h]["xg_by_phase"]) | set(stats[a]["xg_by_phase"]))
    for ph in phases:
        lines.append(f"| {ph} | {stats[h]['xg_by_phase'].get(ph, 0)} | {stats[a]['xg_by_phase'].get(ph, 0)} |")
    lines += ["", "![Shot map](shot_map.png)", "", "![xG race](xg_race.png)", "",
              "![Final-third entries by lane](entries_by_lane.png)", "", "## Set pieces", "",
              md_table(pd.DataFrame(sp_rows)), "", "## Key players", ""]
    for t in teams:
        lines += [f"### {t}", "", md_table(players[t].head(8)), ""]
    big = shots.sort_values("xg", ascending=False).head(8)
    lines += ["## Biggest chances", "",
              md_table(big[["clock", "team", "player", "action", "phase", "xg", "goal", "on_target", "assister"]]), "",
              "<!-- NARRATIVE: tactical analysis and takeaways go here -->", "",
              "---", "_xG from the repo's own model (analysis/xg_model.json). Definitions: see analysis/README.md._"]
    (out / "report.md").write_text("\n".join(lines))
    return {"dir": str(out.relative_to(REPO)), **data["match"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("teams", nargs="*", help="one or two team names (loose matching)")
    ap.add_argument("--date", help="YYYY-MM-DD")
    ap.add_argument("--match-id", type=int)
    args = ap.parse_args()
    if not args.teams and args.match_id is None:
        ap.error("give team name(s) or --match-id")
    match = find_match(*args.teams, date=args.date, match_id=args.match_id)
    print(json.dumps(build(match), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
