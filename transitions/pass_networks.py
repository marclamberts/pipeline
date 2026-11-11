"""
Transition pass networks - 3. Liga 2026-27 (Impect event data)

Pass networks built ONLY from attacking transitions (same definition as
transition_model.py). No xT, no predictions: nodes, edges and every metric
are counts, metres and shares of what happened.

  Nodes  - players; position = mean location of their actions in transitions
  Edges  - completed passes passer -> receiver inside transitions
  Outlet - the first completed pass after a regain (regainer -> outlet player)

Usage:
    python3 transitions/pass_networks.py ["3. Liga 2026-2027"] [--match 277554]
Outputs in transitions/output/:
    pass_network_nodes.csv, pass_network_edges.csv, pass_network_teams.csv,
    pass_networks.html (interactive, self-contained)
"""

import json
import os
import sys
from collections import defaultdict

# Optional single-match mode: --match <impect match id>. Stripped before
# transition_model reads sys.argv for the league folder.
MATCH_ID = None
if "--match" in sys.argv:
    i = sys.argv.index("--match")
    MATCH_ID = int(sys.argv[i + 1])
    del sys.argv[i:i + 2]

import numpy as np
import pandas as pd

import transition_model as tm


def short_name(full):
    parts = (full or "?").split()
    return parts[-1] if parts else "?"


def collect(matches, names):
    nodes = defaultdict(lambda: {"x": [], "y": [], "involvements": 0, "passes": 0, "received": 0,
                                 "outlet_received": 0, "outlet_given": 0, "regains": 0,
                                 "fwd_metres": 0.0, "attempts": 0})
    edges = defaultdict(lambda: {"count": 0, "outlet": 0, "fwd_metres": [], "forward": 0})
    team_games = defaultdict(set)
    team_tr = defaultdict(int)
    player_info = {}

    for f, ev in matches:
        meta = tm.parse_filename(f)
        for run in tm.find_runs(ev):
            res = tm.describe_run(run, ev, meta, names)
            if res is None:
                continue
            row, chain = res
            team = row["team"]
            team_games[team].add(row["match_id"])
            team_tr[team] += 1
            if row["regain_player_id"] is not None:
                nodes[(team, row["regain_player_id"])]["regains"] += 1

            first_pass_done = False
            for e in chain:
                p = e.get("player")
                if not p:
                    continue
                key = (team, p["id"])
                player_info[key] = (p["name"], p.get("position"))
                n = nodes[key]
                n["involvements"] += 1
                c = tm.xy(e)
                if c:
                    n["x"].append(c[0])
                    n["y"].append(c[1])
                if e["actionType"] != "PASS":
                    continue
                n["attempts"] += 1
                rec = ((e.get("pass") or {}).get("receiver") or {})
                if e["result"] != "SUCCESS" or rec.get("type") != "TEAMMATE" or not rec.get("playerId"):
                    first_pass_done = True  # an incomplete first pass ends the outlet window
                    continue
                rid = rec["playerId"]
                end = ((e.get("end") or {}).get("adjCoordinates") or {})
                fwd = (end["x"] - c[0]) if (c and end.get("x") is not None) else 0.0
                ed = edges[(team, p["id"], rid)]
                ed["count"] += 1
                ed["fwd_metres"].append(fwd)
                ed["forward"] += fwd > 5
                n["passes"] += 1
                n["fwd_metres"] += max(0.0, fwd)
                nodes[(team, rid)]["received"] += 1
                if not first_pass_done:
                    ed["outlet"] += 1
                    n["outlet_given"] += 1
                    nodes[(team, rid)]["outlet_received"] += 1
                    first_pass_done = True
    return nodes, edges, team_games, team_tr, player_info


def build_tables(nodes, edges, team_games, team_tr, player_info):
    nrows = []
    for (team, pid), n in nodes.items():
        name, pos = player_info.get((team, pid), (None, None))
        if name is None:
            continue  # receiver who never acted in a transition chain
        nrows.append({
            "team": team, "player_id": pid, "player": name, "short": short_name(name),
            "position": pos,
            "x": round(float(np.mean(n["x"])), 1) if n["x"] else np.nan,
            "y": round(float(np.mean(n["y"])), 1) if n["y"] else np.nan,
            "involvements": n["involvements"], "regains": n["regains"],
            "passes_completed": n["passes"], "pass_attempts": n["attempts"],
            "passes_received": n["received"],
            "outlet_received": n["outlet_received"], "outlet_given": n["outlet_given"],
            "fwd_metres_passing": round(n["fwd_metres"], 0),
        })
    ndf = pd.DataFrame(nrows)
    erows = []
    for (team, a, b), e in edges.items():
        erows.append({"team": team, "from_id": a, "to_id": b, "passes": e["count"],
                      "outlet_passes": e["outlet"],
                      "avg_fwd_m": round(float(np.mean(e["fwd_metres"])), 1),
                      "pct_forward": round(e["forward"] / e["count"], 3)})
    edf = pd.DataFrame(erows)
    names = ndf.set_index(["team", "player_id"])["player"]
    edf["from"] = [names.get((t, i)) for t, i in zip(edf.team, edf.from_id)]
    edf["to"] = [names.get((t, i)) for t, i in zip(edf.team, edf.to_id)]
    edf = edf.dropna(subset=["from", "to"])

    trows = []
    for team, g in ndf.groupby("team"):
        e = edf[edf.team == team]
        total = int(e.passes.sum())
        outlets = int(e.outlet_passes.sum())
        attempts = int(g.pass_attempts.sum())
        top_out = g.sort_values("outlet_received", ascending=False)
        hub = g.assign(touch=g.passes_completed + g.passes_received).sort_values("touch", ascending=False)
        pair = e.sort_values("passes", ascending=False).iloc[0]
        games = len(team_games[team])
        trows.append({
            "team": team, "games": games, "transitions": team_tr[team],
            "transition_passes": total, "passes_pg": round(total / games, 1),
            "completion": round(total / attempts, 3) if attempts else np.nan,
            "passes_per_transition": round(total / team_tr[team], 2),
            "pct_forward": round(float((e.pct_forward * e.passes).sum() / total), 3) if total else np.nan,
            "outlets": outlets,
            "top_outlet": top_out.iloc[0].player,
            "top_outlet_share": round(top_out.iloc[0].outlet_received / outlets, 3) if outlets else np.nan,
            "top2_outlet_share": round(top_out.head(2).outlet_received.sum() / outlets, 3) if outlets else np.nan,
            "hub": hub.iloc[0].player,
            "hub_share": round((hub.iloc[0].touch) / (2 * total), 3) if total else np.nan,
            "top_pair": f"{pair['from']} → {pair['to']}", "top_pair_passes": int(pair.passes),
        })
    tdf = pd.DataFrame(trows).sort_values("passes_pg", ascending=False)
    return ndf, edf, tdf


def page_payload(ndf, edf, tdf, league, n_matches, date_range):
    teams = {}
    for team, g in ndf.groupby("team"):
        g = g.dropna(subset=["x", "y"])
        e = edf[edf.team == team]
        teams[team] = {
            "summary": tdf[tdf.team == team].iloc[0].to_dict(),
            "nodes": g[["player_id", "player", "short", "position", "x", "y", "involvements", "regains",
                        "passes_completed", "passes_received", "outlet_received", "outlet_given",
                        "fwd_metres_passing"]].to_dict("records"),
            "edges": e[["from_id", "to_id", "passes", "outlet_passes", "avg_fwd_m", "pct_forward"]]
            .to_dict("records"),
        }
    return {"league": league, "matches": n_matches, "dates": date_range, "teams": teams}


def to_builtin(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else round(float(o), 3)
    if isinstance(o, float) and np.isnan(o):
        return None
    raise TypeError(type(o))


def main():
    matches = tm.load_matches()
    names = tm.squad_names(matches)  # resolve ids on the full season first
    out_dir = tm.OUT_DIR
    if MATCH_ID is not None:
        matches = [(f, ev) for f, ev in matches if tm.parse_filename(f)[3] == MATCH_ID]
        if not matches:
            sys.exit(f"match {MATCH_ID} not found or has no data")
        out_dir = os.path.join(tm.OUT_DIR, f"match_{MATCH_ID}")
    nodes, edges, team_games, team_tr, info = collect(matches, names)
    ndf, edf, tdf = build_tables(nodes, edges, team_games, team_tr, info)

    os.makedirs(out_dir, exist_ok=True)
    ndf.sort_values(["team", "involvements"], ascending=[True, False]) \
        .to_csv(os.path.join(out_dir, "pass_network_nodes.csv"), index=False)
    edf.sort_values(["team", "passes"], ascending=[True, False]) \
        .to_csv(os.path.join(out_dir, "pass_network_edges.csv"), index=False)
    tdf.to_csv(os.path.join(out_dir, "pass_network_teams.csv"), index=False)

    dates = sorted(tm.parse_filename(f)[0] for f, _ in matches)
    league = os.path.basename(tm.LEAGUE_DIR)
    if MATCH_ID is not None:
        _, home, away, _ = tm.parse_filename(matches[0][0])
        league = f"{league} · {home} vs {away}"
    payload = page_payload(ndf, edf, tdf, league, len(matches), [dates[0], dates[-1]])
    payload["min_default"] = 1 if MATCH_ID is not None else 3
    tpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pass_networks_template.html")
    with open(tpl_path) as fh:
        html = fh.read()
    html = html.replace("/*__DATA__*/null", json.dumps(payload, default=to_builtin, ensure_ascii=False))
    with open(os.path.join(out_dir, "pass_networks.html"), "w") as fh:
        fh.write(html)
    print(tdf.to_string(index=False))


if __name__ == "__main__":
    main()
