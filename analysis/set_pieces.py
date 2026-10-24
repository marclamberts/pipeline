"""Set-piece / restart report for a team, attacking and defending, with league ranks.

    python -m analysis.set_pieces Meppen [--last 10] [--before 2026-11-01]
    python -m analysis.set_pieces Meppen --match-id 277553      # a single match

Writes reports/set_pieces/<team>_<date>/ with report.md, stats.json, set_pieces.csv and PNG charts.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from analysis.footy import pitch, setpieces
from analysis.footy.cache import bundle
from analysis.footy.data import REPO, find_match, find_team, match_index, side_ids, team_matches
from analysis.footy.metrics import md_table

RANKED = [
    ("attacking", "set_piece_xg_per_match", "Set-piece xG for / match"),
    ("defending", "set_piece_xg_per_match", "Set-piece xG against / match"),
    ("attacking", "xg_per_corner", "xG per corner (for)"),
    ("defending", "xg_per_corner", "xG per corner (against)"),
    ("attacking", "corner_first_contact_won_pct", "First contact won, own corners %"),
    ("defending", "corner_first_contact_won_pct", "Opp. first contact won on their corners % (high = bad)"),
    ("attacking", "corners_per_match", "Corners won / match"),
]


def _team_sp(m, team):
    df, shots, sp = bundle(m)
    sp = sp.copy()
    us, _ = side_ids(df, team)
    opp = m.away if m.home == team else m.home
    sp["team"] = np.where(sp.squad_id == us, team, opp)
    sp["match"] = m.label
    sp["match_id"] = m.match_id
    return sp, shots, us


def league_ranks(league: str, before: str | None) -> pd.DataFrame:
    per_team: dict[str, list] = {}
    for m in match_index():
        if m.league != league or (before and m.date >= before):
            continue
        df, _, sp = bundle(m)
        for t in (m.home, m.away):
            us, _ = side_ids(df, t)
            s = sp.copy()
            s["team"] = np.where(s.squad_id == us, t, "opp")
            per_team.setdefault(t, []).append(s)
    rows = []
    for t, parts in per_team.items():
        summ = setpieces.summarise(pd.concat(parts, ignore_index=True), t, len(parts))
        rows.append({"team": t, **{f"{side}.{k}": summ[side][k] for side, k, _ in RANKED}})
    return pd.DataFrame(rows).set_index("team")


def build(team: str, last: int | None, before: str | None, match_id: int | None) -> dict:
    matches = [find_match(match_id=match_id)] if match_id else team_matches(team, last=last, before=before)
    if not matches or team not in (matches[0].home, matches[0].away):
        raise SystemExit(f"No played matches for {team} with those filters.")
    league = matches[-1].league
    parts, sf, sa = [], [], []
    for m in matches:
        sp, shots, us = _team_sp(m, team)
        parts.append(sp)
        sp_shots = shots[shots.set_piece | shots.penalty]
        sf.append(sp_shots[sp_shots.squad_id == us])
        sa.append(sp_shots[sp_shots.squad_id != us])
    sp = pd.concat(parts, ignore_index=True)
    n = len(matches)
    summ = setpieces.summarise(sp, team, n)

    ranks = {}
    if n > 1:
        table = league_ranks(league, before)
        for side, k, label in RANKED:
            col = table[f"{side}.{k}"].dropna()
            v = summ[side][k]
            if v is not None:
                ranks[label] = {"value": v, "league_avg": round(float(col.mean()), 3),
                                "rank": int((col > v).sum() + 1), "of": int(len(col))}

    slug = re.sub(r"\W+", "_", team)
    tag = f"match_{match_id}" if match_id else matches[-1].date
    out = REPO / "reports" / "set_pieces" / f"{slug}_{tag}"
    out.mkdir(parents=True, exist_ok=True)

    def corner_map(rows: pd.DataFrame, path, title: str):
        # Mirror right-side corners onto the left so every delivery reads the same way.
        rows = rows[rows.end_x.notna()].copy()
        flip = rows.side == "right"
        rows.loc[flip, ["y", "end_y"]] *= -1
        rows["good"] = (rows.first_contact == "TEAMMATE") | (rows.shots > 0)
        pitch.event_map(rows, path, title, pitch.TEAM_COLORS[0], half=True, arrows=True, outcome_col="good",
                        subtitle=f"{len(rows)} corners, right-side corners mirrored to the left · green = first contact won or shot")

    att_c = sp[(sp.team == team) & (sp.type == "CORNER")]
    def_c = sp[(sp.team != team) & (sp.type == "CORNER")]
    corner_map(att_c, out / "corners_for.png", f"{team} — corner deliveries")
    corner_map(def_c, out / "corners_against.png", f"{team} — opponent corner deliveries")
    sf, sa = pd.concat(sf), pd.concat(sa)
    pitch.single_team_shot_map(sf, pitch.TEAM_COLORS[0], out / "sp_shots_for.png", f"{team} — set-piece shots")
    pitch.single_team_shot_map(sa, pitch.TEAM_COLORS[1], out / "sp_shots_against.png", f"{team} — set-piece shots conceded")

    sp["sp_id"] = sp.sp_id.astype("Int64")
    cols = ["match", "match_id", "sp_id", "clock", "type", "team", "taker", "foot", "side", "short", "swing", "target_zone",
            "first_contact", "long_throw", "shots", "xg", "goals", "shooters", "counter_shots_conceded"]
    sp[cols].round(3).to_csv(out / "set_pieces.csv", index=False)

    zones = lambda d: ", ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])) or "–"
    data = {"team": team, "league": league, "matches": [m.label for m in matches], "summary": summ, "league_ranks": ranks}
    (out / "stats.json").write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str))

    A, D = summ["attacking"], summ["defending"]
    lines = [f"# Set-piece report — {team}", "",
             f"_{league} · {n} match{'es' if n > 1 else ''}: {matches[0].date} → {matches[-1].date}_", "",
             "<!-- NARRATIVE: headline findings and set-piece game plan -->", "",
             "## Overview", "", "| | attacking | defending |", "|---|---|---|"]
    for k, label in [("set_piece_shots", "Set-piece shots"), ("set_piece_xg", "Set-piece xG"),
                     ("set_piece_xg_per_match", "Set-piece xG / match"), ("set_piece_goals", "Set-piece goals (incl. pens)"),
                     ("penalties", "Penalties"), ("counter_shots_conceded_after", "Counter shots conceded after (≤15 s)"),
                     ("counter_goals_conceded_after", "Counter goals conceded after")]:
        lines.append(f"| {label} | {A[k]} | {D[k]} |")
    if ranks:
        lines += ["", "### League context", "", "| metric | value | league avg | rank |", "|---|---|---|---|"]
        lines += [f"| {k} | {r['value']} | {r['league_avg']} | {r['rank']}/{r['of']} |" for k, r in ranks.items()]
    lines += ["", "## Corners", "", "| | attacking | defending |", "|---|---|---|"]
    for k, label in [("corners", "Corners"), ("corners_per_match", "Per match"), ("corner_short_pct", "Short %"),
                     ("corner_inswing_pct", "Inswing % (delivered)"), ("corner_first_contact_won_pct", "First contact won %"),
                     ("corner_shots", "Shots"), ("corner_xg", "xG"), ("xg_per_corner", "xG per corner"),
                     ("corner_goals", "Goals")]:
        lines.append(f"| {label} | {A[k]} | {D[k]} |")
    lines += [f"| Target zones | {zones(A['corner_target_zones'])} | {zones(D['corner_target_zones'])} |", "",
              "![Corners for](corners_for.png) ![Corners against](corners_against.png)", "",
              "## Free kicks, throw-ins, goal kicks", "", "| | attacking | defending |", "|---|---|---|"]
    for k, label in [("attacking_free_kicks", "Free kicks in final third"), ("direct_fk_shots", "Direct FK shots"),
                     ("fk_xg", "FK xG"), ("fk_goals", "FK goals"), ("throw_ins", "Throw-ins"),
                     ("long_throws", "Long throws into the box area"), ("long_throw_xg", "Long-throw xG"),
                     ("goal_kicks", "Goal kicks"), ("goal_kick_short_pct", "Goal kicks short %")]:
        lines.append(f"| {label} | {A[k]} | {D[k]} |")
    lines += ["", "![Set-piece shots](sp_shots_for.png) ![Set-piece shots conceded](sp_shots_against.png)", "",
              "## People", "", "**Takers**", "", md_table(pd.DataFrame(summ["takers"])), "",
              "**Set-piece shooters (targets)**", "", md_table(pd.DataFrame(summ["targets"])), "",
              "## Every dangerous set piece", ""]
    danger = sp[(sp.type != "GOAL_KICK") & ((sp.shots > 0) | (sp.counter_shots_conceded.fillna(0) > 0))].copy()
    danger["counter_shots_conceded"] = danger.counter_shots_conceded.fillna(0).astype(int)
    lines += [md_table(danger[["match", "clock", "type", "team", "taker", "target_zone", "first_contact",
                               "shots", "xg", "goals", "shooters", "counter_shots_conceded"]].round(2)), "",
              "<!-- NARRATIVE: routines spotted, matchups, recommendations -->", "", "---",
              "_Set-piece shots = shots in the set-piece sequence or ≤10 s after it, plus penalties. "
              "Full list of restarts: set_pieces.csv. Definitions: analysis/README.md._"]
    (out / "report.md").write_text("\n".join(lines))
    return {"dir": str(out.relative_to(REPO)), "team": team, "matches": n}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("team")
    ap.add_argument("--last", type=int, default=None, help="last N matches (default: whole season so far)")
    ap.add_argument("--before", help="only matches before this date (YYYY-MM-DD)")
    ap.add_argument("--match-id", type=int)
    args = ap.parse_args()
    print(json.dumps(build(find_team(args.team), args.last, args.before, args.match_id), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
