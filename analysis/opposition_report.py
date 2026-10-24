"""Opposition report: how a team plays, aggregated over its recent matches,
with league ranks for context.

    python -m analysis.opposition_report Meppen [--last 5] [--before 2026-11-01] [--venue home|away|all]

Writes reports/opposition/<team>_<date>/ with report.md, stats.json and PNG charts.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from analysis.footy import pitch, setpieces
from analysis.footy.cache import bundle
from analysis.footy.data import REPO, find_team, match_index, score, side_ids, team_matches
from analysis.footy.metrics import LANES, md_table, player_table, team_stats

# Metrics averaged per match and ranked in the league (True = higher is "more").
RANKED = [
    ("xg", "xG for"), ("xg_against", "xG against"), ("shots", "Shots"), ("xg_per_shot", "xG per shot"),
    ("possession_pct", "Possession % (pass share)"), ("pass_accuracy_pct", "Pass accuracy %"),
    ("long_pass_pct", "Long passes %"), ("progressive_passes", "Progressive passes"),
    ("final_third_entries", "Final-third entries"), ("box_entries", "Box entries"), ("crosses", "Crosses"),
    ("field_tilt_pct", "Field tilt %"), ("ppda", "PPDA (low = intense press)"),
    ("high_regains", "High regains"), ("own_half_losses", "Ground losses in own half"),
    ("pressure_on_opp_buildup", "Pressure on opp. build-up"), ("set_piece_xg", "Set-piece-phase xG (all restarts)"),
]


def _match_bundle(m):
    df, shots, _ = bundle(m)
    return df, shots


def league_table(league: str, before: str | None) -> pd.DataFrame:
    """Average per-match team stats for every team in the league (season to date)."""
    rows = []
    for m in match_index():
        if m.league != league or (before and m.date >= before):
            continue
        df, shots = _match_bundle(m)
        for t in (m.home, m.away):
            st = team_stats(df, t, shots)
            rows.append({"team": t, **{k: st[k] for k, _ in RANKED}})
    return pd.DataFrame(rows).groupby("team").mean(numeric_only=True)


def build(team: str, last: int, before: str | None, venue: str) -> dict:
    matches = team_matches(team, last=last, before=before, venue=venue)
    if not matches:
        raise SystemExit(f"No played matches for {team} with those filters.")
    league = matches[-1].league

    results, stats, all_shots, players, sps, own_losses, opp_entries = [], [], [], [], [], [], []
    build_up = {"gk_short": 0, "gk_total": 0, "own_third_passes": 0, "own_third_pressured": 0,
                "own_third_pressured_ok": 0}
    for m in matches:
        df, shots = _match_bundle(m)
        us, them = side_ids(df, team)
        opp = m.away if m.home == team else m.home
        st = team_stats(df, team, shots)
        ost = team_stats(df, opp, shots)
        sc = score(df)
        stats.append(st)
        opp_entries.append(ost["entries_by_lane"])
        results.append({
            "date": m.date, "opponent": opp, "venue": "H" if m.home == team else "A",
            "score": f"{sc[team]}-{sc[opp]}",
            "result": "W" if sc[team] > sc[opp] else ("D" if sc[team] == sc[opp] else "L"),
            "xg": st["xg"], "xg_against": st["xg_against"], "formation": st["formation"],
            "opp_formation": ost["formation"],
        })
        s = shots.copy()
        s["for"] = s.squad_id == us
        s["match"] = m.date
        all_shots.append(s)
        p = player_table(df, team, shots)
        p["minutes"] = (p.last_min - p.first_min).clip(lower=1)
        players.append(p)
        sp = bundle(m)[2].copy()
        sp["team"] = np.where(sp.squad_id == us, team, opp)
        sps.append(sp)

        mine = df[df.squad_id == us]
        lost = mine[mine.action_type.isin(["PASS", "DRIBBLE"]) & (mine.result == "FAIL")
                    & mine.action.isin(["LOW_PASS", "DRIBBLE"]) & (mine.x < 0)]
        own_losses.append(lost[["x", "y"]])
        gk = mine[mine.action_type == "GOAL_KICK"]
        build_up["gk_total"] += len(gk)
        build_up["gk_short"] += int((gk.pass_dist < 30).sum())
        own3 = mine[(mine.action_type == "PASS") & (mine.x < -17.5)]
        press = own3[own3.pressure >= 50]
        build_up["own_third_passes"] += len(own3)
        build_up["own_third_pressured"] += len(press)
        build_up["own_third_pressured_ok"] += int((press.result == "SUCCESS").sum())

    n = len(matches)
    shots = pd.concat(all_shots, ignore_index=True)
    sf, sa = shots[shots["for"]], shots[~shots["for"]]
    avg = pd.DataFrame(stats)[[k for k, _ in RANKED]].mean().round(2)

    table = league_table(league, before)
    ranks = {}
    for k, label in RANKED:
        col = table[k].dropna()
        rank = int((col > avg[k]).sum() + 1)
        ranks[k] = {"label": label, "value": float(avg[k]), "league_avg": round(float(col.mean()), 2),
                    "rank": rank, "of": int(len(col))}

    lanes_for = {l: round(np.mean([s["entries_by_lane"][l] for s in stats]), 1) for l in LANES}
    # Opponents' lanes are in their own frame; mirror so they name *our* flank (their left = our right).
    mirror = dict(zip(LANES, LANES[::-1]))
    lanes_against = {l: round(np.mean([e[mirror[l]] for e in opp_entries]), 1) for l in LANES}
    phase_for = (sf.groupby("phase").xg.sum() / n).round(2).to_dict()
    phase_against = (sa.groupby("phase").xg.sum() / n).round(2).to_dict()

    pl = pd.concat(players).groupby(["player"], as_index=False).agg(
        position=("position", lambda s: s.mode().iat[0] if s.notna().any() else None),
        apps=("player", "size"), minutes=("minutes", "sum"), goals=("goals", "sum"), shots=("shots", "sum"),
        xg=("xg", "sum"), key_passes=("key_passes", "sum"), xa=("xa", "sum"), prog_passes=("prog_passes", "sum"),
        box_entries=("box_entries", "sum"), regains=("regains", "sum"), losses=("losses", "sum"))
    pl["xg+xa_p90"] = ((pl.xg + pl.xa) * 90 / pl.minutes.clip(lower=90)).round(2)
    pl = pl.round(2).sort_values(["minutes"], ascending=False)
    key_players = pl[pl.minutes >= 0.4 * 90 * n].sort_values("xg+xa_p90", ascending=False).head(10)

    sp_all = pd.concat(sps, ignore_index=True)
    sp_sum = setpieces.summarise(sp_all, team, n)

    slug = re.sub(r"\W+", "_", team)
    out = REPO / "reports" / "opposition" / f"{slug}_{matches[-1].date}"
    out.mkdir(parents=True, exist_ok=True)
    pitch.single_team_shot_map(sf, pitch.TEAM_COLORS[0], out / "shots_for.png", f"{team} — shots for (last {n})")
    pitch.single_team_shot_map(sa, pitch.TEAM_COLORS[1], out / "shots_against.png", f"{team} — shots conceded (last {n})")
    pitch.lane_bars(lanes_for, out / "entries_by_lane.png", f"{team} — final-third entries by lane",
                    against=lanes_against, labels=(f"{team} attacking", "opponents attacking"),
                    subtitle=f"lanes named from {team}'s side: Left Wing = {team}'s left flank, both ways")
    pitch.event_map(pd.concat(own_losses), out / "own_half_losses.png", f"{team} — ground losses in own half",
                    pitch.TEAM_COLORS[1], subtitle=f"failed ground passes/dribbles, last {n} matches, attacking →")

    gk_short = round(100 * build_up["gk_short"] / max(1, build_up["gk_total"]), 1)
    press_ok = round(100 * build_up["own_third_pressured_ok"] / max(1, build_up["own_third_pressured"]), 1)
    form = pd.Series([r["formation"] for r in results]).value_counts().to_dict()
    data = {
        "team": team, "league": league, "matches": n, "venue_filter": venue, "results": results,
        "formations": form, "averages_and_ranks": ranks,
        "entries_by_lane_for": lanes_for, "entries_by_lane_against": lanes_against,
        "xg_by_phase_for": phase_for, "xg_by_phase_against": phase_against,
        "build_up": {"goal_kicks_short_pct": gk_short,
                     "own_third_passes_under_pressure_pct": round(100 * build_up["own_third_pressured"] / max(1, build_up["own_third_passes"]), 1),
                     "own_third_pressured_pass_success_pct": press_ok},
        "key_players": key_players.to_dict("records"), "set_pieces": sp_sum,
    }
    (out / "stats.json").write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str))

    rec = pd.Series([r["result"] for r in results]).value_counts()
    lines = [f"# Opposition report — {team}", "",
             f"_{league} · last {n} matches{'' if venue == 'all' else f' ({venue})'} · "
             f"{matches[0].date} → {matches[-1].date} · W{rec.get('W', 0)} D{rec.get('D', 0)} L{rec.get('L', 0)}_", "",
             "<!-- NARRATIVE: executive summary (how they play, 3 strengths, 3 weaknesses, how to beat them) -->", "",
             "## Results", "", md_table(pd.DataFrame(results)), "",
             f"**Formations used:** {', '.join(f'{k} ×{v}' for k, v in form.items())}", "",
             "## Profile vs league", "",
             f"Rank 1 = highest value among {ranks['xg']['of']} teams (for PPDA, losses and xG against, high is bad).", "",
             "| metric | per match | league avg | rank |", "|---|---|---|---|"]
    for k, r in ranks.items():
        lines.append(f"| {r['label']} | {r['value']} | {r['league_avg']} | {r['rank']}/{r['of']} |")
    lines += ["", "## In possession", "",
              f"- Goal kicks played short (<30 m): **{gk_short}%**",
              f"- Own-third passes made under pressure (≥50): **{data['build_up']['own_third_passes_under_pressure_pct']}%**, "
              f"completed **{press_ok}%** of those", "",
              "### xG per match by phase", "", "| phase | for | against |", "|---|---|---|"]
    for ph in sorted(set(phase_for) | set(phase_against)):
        lines.append(f"| {ph} | {phase_for.get(ph, 0)} | {phase_against.get(ph, 0)} |")
    lines += ["", "![Final-third entries by lane](entries_by_lane.png)", "",
              "![Shots for](shots_for.png) ![Shots against](shots_against.png)", "",
              "## Out of possession", "",
              f"- PPDA **{ranks['ppda']['value']}** (league {ranks['ppda']['league_avg']}), "
              f"high regains **{ranks['high_regains']['value']}**/match, pressure on opp. build-up "
              f"**{ranks['pressure_on_opp_buildup']['value']}**",
              f"- Opponents' final-third entries by {team} flank (mirrored, per match): "
              + ", ".join(f"{l.replace('_', ' ').lower()} {v}" for l, v in lanes_against.items()), "",
              "![Own-half losses](own_half_losses.png)", "",
              "## Set pieces", "",
              "| | attacking | defending |", "|---|---|---|"]
    for k in ["corners_per_match", "corner_short_pct", "corner_inswing_pct", "corner_first_contact_won_pct",
              "corner_xg", "xg_per_corner", "corner_goals", "attacking_free_kicks", "fk_xg", "long_throws",
              "penalties", "set_piece_xg_per_match", "set_piece_goals"]:
        lines.append(f"| {k} | {sp_sum['attacking'][k]} | {sp_sum['defending'][k]} |")
    lines += ["", "Takers: " + ", ".join(f"{t['taker']} ({t['type'].lower()}, {str(t['foot']).lower()}, {t['n']})" for t in sp_sum["takers"][:4]),
              "Main set-piece shooters: " + ", ".join(f"{t['player']} ({t['set_piece_shots']})" for t in sp_sum["targets"][:5]),
              "", "## Key players", "", "_Minutes are estimated from first/last event._", "",
              md_table(key_players), "",
              "<!-- NARRATIVE: game plan recommendations go here -->", "", "---",
              "_xG from the repo's own model. Definitions: analysis/README.md._"]
    (out / "report.md").write_text("\n".join(lines))
    return {"dir": str(out.relative_to(REPO)), "team": team, "matches": n, "league": league}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("team")
    ap.add_argument("--last", type=int, default=5)
    ap.add_argument("--before", help="only matches before this date (YYYY-MM-DD)")
    ap.add_argument("--venue", choices=["all", "home", "away"], default="all")
    args = ap.parse_args()
    print(json.dumps(build(find_team(args.team), args.last, args.before, args.venue), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
