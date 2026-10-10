"""Extract goal-kick possessions and all shots from Impect event JSON files.

Outputs (in analysis/output/):
  shots.csv       - every shot, with features for the xG model
  goalkicks.csv   - one row per goal kick, with routine descriptors + outcomes
  teams.csv       - squadId -> team name mapping

Coordinates: adjCoordinates, attacking towards +x, pitch 105x68 centred at 0,
y > 0 is the LEFT side of the team in possession.
"""
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict
from multiprocessing import Pool

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "output")

POS_SHORT = {
    "GOALKEEPER": "GK",
    "CENTRAL_DEFENDER": "CB",
    "LEFT_WINGBACK_DEFENDER": "LB",
    "RIGHT_WINGBACK_DEFENDER": "RB",
    "DEFENSE_MIDFIELD": "DM",
    "CENTRAL_MIDFIELD": "CM",
    "ATTACKING_MIDFIELD": "AM",
    "LEFT_WINGER": "LW",
    "RIGHT_WINGER": "RW",
    "CENTER_FORWARD": "CF",
}


def g(e, *keys):
    for k in keys:
        if not isinstance(e, dict):
            return None
        e = e.get(k)
    return e


def side(y):
    if y is None:
        return None
    if y > 11:
        return "L"
    if y < -11:
        return "R"
    return "C"


def gk_zone(x, y):
    """Where the goal kick (first pass) ends up."""
    if x is None:
        return "unknown"
    if x <= -36 and abs(y) <= 20.16:
        return "short_box"
    if x < -17.5:
        return "short_wide" if abs(y) > 20.16 else "short_edge"
    if x < 0:
        return "mid"
    return "long"


# File names replace non-ASCII letters with "_"; restore the ones in these leagues.
NAME_FIXES = {
    "K_ln": "Köln", "M_nchengladbach": "Mönchengladbach", "M_nchen": "München", "F_rth": "Fürth",
    "D_sseldorf": "Düsseldorf", "W_rzburger": "Würzburger", "N_rnberg": "Nürnberg",
    "Osnabr_ck": "Osnabrück", "Saarbr_cken": "Saarbrücken", "Preu_en": "Preußen", "M_nster": "Münster",
    "Gro_aspach": "Großaspach",
}


def fix_name(name):
    for bad, good in NAME_FIXES.items():
        name = name.replace(bad, good)
    return name.replace("_", " ")


def parse_match(path):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    if not d:
        return None
    league = path.split(os.sep)[-2].replace(" 2026-2027", "")
    fname = os.path.basename(path)
    m = re.match(r"(\d{4}-\d{2}-\d{2})_(.+)_vs_(.+)_(\d+)\.json", fname)
    date, home, away, match_id = m.groups()
    squads = sorted({e["squadId"] for e in d if e["squadId"] is not None})

    shots = []
    for i, e in enumerate(d):
        if e["actionType"] != "SHOT":
            continue
        nxt = d[i + 1]["actionType"] if i + 1 < len(d) else None
        sx, sy = g(e, "start", "adjCoordinates", "x"), g(e, "start", "adjCoordinates", "y")
        if sx is None:
            continue
        dx = 52.5 - sx
        dist = math.hypot(dx, sy)
        # opening angle to the goal mouth (posts at y=+-3.66)
        a = math.atan2(7.32 * dx, dx * dx + sy * sy - 3.66 ** 2)
        if a < 0:
            a += math.pi
        shots.append(
            dict(
                league=league, match_id=match_id, idx=e["index"], squad=e["squadId"],
                x=sx, y=sy, dist=dist, angle=a, action=e["action"], phase=e["phase"],
                body=e["bodyPart"], pressure=e["pressure"] or 0.0, opponents=e["opponents"],
                goal=int(e["result"] == "SUCCESS" and nxt == "GOAL"),
            )
        )

    gks = []
    n = len(d)
    for i, e in enumerate(d):
        if e["actionType"] != "GOAL_KICK":
            continue
        team = e["squadId"]
        opp = next((s for s in squads if s != team), None)
        t0 = e["gameTime"]["gameTimeInSec"]
        period = e["periodId"]
        seq = e["sequenceIndex"]

        # Possession chain: same sequence, plus following sequences that the
        # kicking team still owns (restarts it won: throw-ins, free kicks, corners).
        chain = []
        cur_seq = seq
        j = i
        while j < n:
            ev = d[j]
            if ev["periodId"] != period or ev["actionType"] in ("FINAL_WHISTLE",):
                break
            if ev["sequenceIndex"] != cur_seq:
                if ev["currentAttackingSquadId"] != team:
                    break
                # a corner / penalty / free kick in the opponent half is its own
                # set-piece situation, not part of the goal-kick routine
                if ev["actionType"] in ("CORNER", "PENALTY") or ev["action"] == "PENALTY_KICK":
                    break
                if ev["actionType"] == "FREE_KICK" and (g(ev, "start", "adjCoordinates", "x") or 0) > 0:
                    break
                cur_seq = ev["sequenceIndex"]
            chain.append(ev)
            if ev["actionType"] in ("GOAL", "OWN_GOAL"):
                break
            j += 1
        same_seq = [ev for ev in chain if ev["sequenceIndex"] == seq]

        ex, ey = g(e, "end", "adjCoordinates", "x"), g(e, "end", "adjCoordinates", "y")
        gk_receiver_type = g(e, "pass", "receiver", "type")
        # who received the goal kick
        rec = next(
            (ev for ev in same_seq[1:3] if ev["actionType"] in ("RECEPTION", "LOOSE_BALL_REGAIN")),
            None,
        )
        rec_pos = POS_SHORT.get(g(rec, "player", "position"), "?") if rec else None
        rec_team = rec["squadId"] if rec else None

        # Passing path of the kicking team in the first possession (own passes only)
        passes = [
            ev for ev in same_seq
            if ev["squadId"] == team and ev["actionType"] in ("PASS", "GOAL_KICK", "CLEARANCE")
            and g(ev, "end", "adjCoordinates", "x") is not None
        ]
        path = []
        for p in passes[:4]:
            path.append(
                (POS_SHORT.get(g(p, "player", "position"), "?"),
                 g(p, "start", "adjCoordinates", "x"), g(p, "start", "adjCoordinates", "y"),
                 g(p, "end", "adjCoordinates", "x"), g(p, "end", "adjCoordinates", "y"),
                 p["action"], p["result"])
            )
        # First "launch": a pass from own third ending in the opponent half
        launch = None
        for k, p in enumerate(passes[:6]):
            sx_, ex_ = g(p, "start", "adjCoordinates", "x"), g(p, "end", "adjCoordinates", "x")
            if sx_ < -17.5 and ex_ >= 0:
                launch = (k, POS_SHORT.get(g(p, "player", "position"), "?"), side(g(p, "end", "adjCoordinates", "y")),
                          p["result"])
                break

        # Intended exit route: side of the first own pass that leaves the box
        exit_route = None
        for p in passes[:6]:
            px, py = g(p, "end", "adjCoordinates", "x"), g(p, "end", "adjCoordinates", "y")
            if px > -36 or abs(py) > 20.16:
                exit_route = side(py)
                break

        # progress milestones (receptions/carries by kicking team in chain)
        def first_time(pred):
            for ev in chain:
                if ev["squadId"] != team:
                    continue
                x = g(ev, "start", "adjCoordinates", "x")
                if x is not None and pred(x, ev):
                    return ev["gameTime"]["gameTimeInSec"] - t0
            return None

        t_mid = first_time(lambda x, ev: x >= -17.5 and ev["actionType"] in ("RECEPTION", "DRIBBLE", "PASS", "LOOSE_BALL_REGAIN"))
        t_final = first_time(lambda x, ev: x >= 17.5 and ev["actionType"] in ("RECEPTION", "DRIBBLE", "PASS", "LOOSE_BALL_REGAIN", "SHOT"))
        t_box = first_time(lambda x, ev: x >= 36 and abs(g(ev, "start", "adjCoordinates", "y") or 99) <= 20.16)

        # first lane the team progressed through when entering middle third
        exit_side = None
        for ev in chain:
            if ev["squadId"] == team and ev["actionType"] in ("RECEPTION", "LOOSE_BALL_REGAIN"):
                x = g(ev, "start", "adjCoordinates", "x")
                if x is not None and x >= -17.5:
                    exit_side = side(g(ev, "start", "adjCoordinates", "y"))
                    break

        # possession end: how/where the chain ended
        last = chain[-1] if chain else e
        end_x = g(last, "start", "adjCoordinates", "x")
        lost_own_third = None
        if j < n and d[j]["currentAttackingSquadId"] != team and d[j]["periodId"] == period:
            # turnover location from kicking team's perspective
            lx = g(d[j], "start", "adjCoordinates", "x")
            lost_own_third = int(lx is not None and lx >= 17.5)  # opponent coords: >=17.5 means our own third

        # Counter-risk: opponent shots in the possession that immediately follows an
        # open-play turnover (not after a stoppage such as our shot going out).
        opp_shot_idx = []
        if j < n and d[j]["periodId"] == period and d[j]["currentAttackingSquadId"] == opp \
                and d[j]["phase"] != "SET_PIECE":
            k, oseq = j, d[j]["sequenceIndex"]
            while k < n and d[k]["periodId"] == period:
                ev = d[k]
                if ev["sequenceIndex"] != oseq:
                    if ev["currentAttackingSquadId"] != opp or ev["actionType"] in ("CORNER",) \
                            or ev["action"] == "PENALTY_KICK" or \
                            (ev["actionType"] == "FREE_KICK" and (g(ev, "start", "adjCoordinates", "x") or 0) > 0):
                        break
                    oseq = ev["sequenceIndex"]
                if ev["actionType"] == "SHOT" and ev["squadId"] == opp:
                    opp_shot_idx.append(ev["index"])
                if ev["actionType"] in ("GOAL", "OWN_GOAL"):
                    break
                k += 1

        chain_shots = [ev for ev in chain if ev["actionType"] == "SHOT" and ev["squadId"] == team]
        shot_idx = [ev["index"] for ev in chain_shots]
        first_shot_t = chain_shots[0]["gameTime"]["gameTimeInSec"] - t0 if chain_shots else None
        shot_in_seq = any(ev["actionType"] == "SHOT" and ev["squadId"] == team for ev in same_seq)
        n_team_passes = sum(1 for ev in chain if ev["squadId"] == team and ev["actionType"] == "PASS")
        restarts = sum(1 for ev in chain[1:] if ev["phase"] == "SET_PIECE" and ev.get("setPiece") and g(ev, "setPiece", "mainEvent"))

        gks.append(
            dict(
                league=league, match_id=match_id, date=date, home=home, away=away,
                team=team, opp=opp, idx=e["index"], period=period, t=t0,
                taker=POS_SHORT.get(g(e, "player", "position"), "?"),
                taker_name=g(e, "player", "name"),
                gk_end_x=ex, gk_end_y=ey, gk_dist=g(e, "pass", "distance"),
                gk_zone=gk_zone(ex, ey), gk_side=side(ey), gk_result=e["result"],
                gk_action=e["action"], gk_receiver_type=gk_receiver_type,
                rec_pos=rec_pos, rec_own=int(rec_team == team) if rec_team is not None else None,
                rec_name=g(rec, "player", "name") if rec else None,
                opp_formation=g(e, "formation", "opponent"), formation=g(e, "formation", "team"),
                path=json.dumps(path),
                path_pos="-".join([p[0] for p in path]),
                launch_k=launch[0] if launch else None,
                launch_by=launch[1] if launch else None,
                launch_side=launch[2] if launch else None,
                launch_result=launch[3] if launch else None,
                n_passes_seq=len(passes), n_team_passes_chain=n_team_passes,
                restarts_won=restarts,
                t_mid=t_mid, t_final=t_final, t_box=t_box, exit_side=exit_side, exit_route=exit_route,
                lost_own_third=lost_own_third,
                chain_dur=(last["gameTime"]["gameTimeInSec"] - t0),
                n_shots=len(chain_shots), shot_idx=json.dumps(shot_idx),
                opp_shot_idx=json.dumps(opp_shot_idx),
                shot_in_seq=int(shot_in_seq), t_first_shot=first_shot_t,
                first_shot_phase=chain_shots[0]["phase"] if chain_shots else None,
            )
        )
    meta = dict(match_id=match_id, home=home, away=away, squads=squads)
    return shots, gks, meta


def main():
    os.makedirs(OUT, exist_ok=True)
    files = [f for f in sorted(glob.glob(os.path.join(ROOT, "*", "*.json"))) if os.path.getsize(f) > 10000]
    with Pool() as pool:
        res = [r for r in pool.map(parse_match, files) if r]
    shots = pd.DataFrame([s for r in res for s in r[0]])
    gks = pd.DataFrame([g_ for r in res for g_ in r[1]])

    # squadId -> name: intersect squad sets over each team's matches
    cand = defaultdict(Counter)
    for _, _, m in res:
        for name in (m["home"], m["away"]):
            for s in m["squads"]:
                cand[name][s] += 1
    sq2name = {}
    for name, c in cand.items():
        (s, cnt), = c.most_common(1)
        sq2name[s] = fix_name(name)
    teams = pd.DataFrame(sorted(sq2name.items()), columns=["squad", "team_name"])
    shots.to_csv(os.path.join(OUT, "shots.csv"), index=False)
    gks.to_csv(os.path.join(OUT, "goalkicks.csv"), index=False)
    teams.to_csv(os.path.join(OUT, "teams.csv"), index=False)
    print(f"matches={len(res)} shots={len(shots)} goalkicks={len(gks)} teams={len(teams)}")


if __name__ == "__main__":
    main()
