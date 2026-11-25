"""PPDA (passes allowed per defensive action) from Opta event JSON.

Reproduces Opta's own `ppda` matchstat exactly (verified 62/62 team-matches):

  numerator   = opponent passes (typeId 1, any outcome) with x <= 60
                (i.e. in the opponent's own 60% of the pitch)
  denominator = defensive actions by the pressing team with x >= 40:
                tackles (7, won or lost), interceptions (8, outcome 1),
                challenges (45, outcome 0), fouls committed (4, outcome 0),
                blocked passes (74, outcome 1)

Only regular/extra-time periods (1-4) count; shootouts and pre/post-match
events are ignored. Opta x is 0-100 in the acting team's attacking direction.

Usage:  python3 ppda.py "Eredivisie 2026-2027" [out.csv]
"""
import csv
import glob
import json
import os
import sys

DEF_ACTIONS = {(7, 0), (7, 1), (8, 1), (45, 0), (4, 0), (74, 1)}
PLAY_PERIODS = {1, 2, 3, 4}


def match_ppda(events):
    teams = sorted({e["contestantId"] for e in events if e.get("contestantId")})
    out = {}
    for me in teams:
        opp = next(t for t in teams if t != me)
        passes = sum(
            1 for e in events
            if e.get("contestantId") == opp and e["typeId"] == 1
            and e["periodId"] in PLAY_PERIODS and e["x"] <= 60
        )
        actions = sum(
            1 for e in events
            if e.get("contestantId") == me and e["periodId"] in PLAY_PERIODS
            and e["x"] >= 40 and (e["typeId"], e.get("outcome")) in DEF_ACTIONS
        )
        out[me] = {
            "opposition_passes": passes,
            "def_actions": actions,
            "ppda": round(passes / actions, 1) if actions else None,
        }
    return out


def main(folder, out_path=None):
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        for team, r in match_ppda(data["event"]).items():
            rows.append({"match": os.path.basename(path)[:-5], "contestantId": team, **r})
    w = csv.DictWriter(open(out_path, "w", newline="") if out_path else sys.stdout,
                       fieldnames=["match", "contestantId", "opposition_passes", "def_actions", "ppda"])
    w.writeheader()
    w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
