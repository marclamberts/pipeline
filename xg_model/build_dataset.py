"""Extract every shot from the league event folders into one flat feature table.

Coordinates use `adjCoordinates`: pitch is 105 x 68, centred on (0, 0), and the
attacking team always shoots towards x = +52.5.
"""
import glob
import json
import math
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "shots.csv")

GOAL_X = 52.5
POST_Y = 7.32 / 2

DIST_TO_OPP = {
    "LESS_THAN_ONE_METER": 0.5, "ONE_METER": 1, "TWO_METERS": 2,
    "THREE_METERS": 3, "FOUR_METERS": 4, "MORE_THAN_FOUR_METERS": 5,
}


def goal_mouth_angle(x, y):
    """Visible angle (degrees) of the goal mouth from the shot location."""
    dx = GOAL_X - x
    a1 = math.atan2(POST_Y - y, dx)
    a2 = math.atan2(-POST_Y - y, dx)
    return math.degrees(abs(a1 - a2))


def find_assist(events, i, shot):
    """Last pass by the shooting team within 4 events, if it reached the shooter."""
    for j in range(i - 1, max(i - 5, -1), -1):
        p = events[j]
        if p["pass"] and p["squadId"] == shot["squadId"]:
            receiver = (p["pass"].get("receiver") or {}).get("playerId")
            if receiver == shot["player"]["id"]:
                return p
            return None
    return None


def set_piece_origin(events, i, restarts):
    """The restart event (corner, free kick, throw-in, ...) that started this set piece."""
    sp = events[i]["setPiece"]
    if not sp:
        return None
    return restarts.get(sp["id"])


def set_piece_features(events, i, restarts, assist):
    e = events[i]
    is_pen = e["action"] == "PENALTY_KICK"
    o = set_piece_origin(events, i, restarts)
    if o is None and not is_pen:
        return {"is_set_piece": 0}
    if is_pen:
        restart = "PENALTY"
    elif o["actionType"] == "SHOT":
        restart = "DIRECT_FREE_KICK" if o["action"] in ("DIRECT_FREE_KICK", "LONG_RANGE_SHOT", "MID_RANGE_SHOT") else o["action"]
    else:
        restart = o["actionType"]
    row = {"is_set_piece": 1, "restart_type": restart}
    if o is None:
        return row
    ox = o["start"]["adjCoordinates"]["x"]
    oy = o["start"]["adjCoordinates"]["y"]
    foot = o["bodyPartExtended"]
    row.update({
        "restart_x": ox,
        "restart_abs_y": abs(oy),
        "restart_dist_to_goal": math.hypot(GOAL_X - ox, oy),
        "time_since_restart": e["gameTime"]["gameTimeInSec"] - o["gameTime"]["gameTimeInSec"],
        "events_since_restart": e["index"] - o["index"],
        # shot is struck from the delivery itself (first contact) or is the restart itself
        "first_contact": int(o is e or (assist is not None and assist["id"] == o["id"])),
        # still in the same sub-phase as the delivery, i.e. not a second ball / recycled attack
        "first_phase": int(o["setPiece"]["subPhaseId"] == e["setPiece"]["subPhaseId"]),
        "delivery_distance": (o["pass"] or {}).get("distance"),
        # taker's foot relative to the side of the pitch: 1 and -1 separate in- and out-swinging deliveries
        "delivery_swing": (1 if foot == "FOOT_RIGHT" else -1 if foot == "FOOT_LEFT" else 0) * (1 if oy >= 0 else -1),
        "short_delivery": int(o["pass"] is not None and o["pass"]["distance"] < 15),
    })
    return row


def shot_row(events, i, league, match_file, restarts):
    e = events[i]
    s = e["shot"]
    x = e["start"]["adjCoordinates"]["x"]
    y = e["start"]["adjCoordinates"]["y"]
    gk = (s.get("gk") or {}).get("adjCoordinates") or {}
    gk_x, gk_y = gk.get("x"), gk.get("y")
    prev = events[i - 1] if i > 0 else None
    assist = find_assist(events, i, e)

    row = {
        "league": league,
        "match_file": match_file,
        "match_id": match_file.rsplit("_", 1)[-1].replace(".json", ""),
        "event_id": e["id"],
        "period": e["periodId"],
        "game_time_sec": e["gameTime"]["gameTimeInSec"],
        "squad_id": e["squadId"],
        "player_id": e["player"]["id"],
        "player_name": e["player"]["name"],
        "player_position": e["player"]["position"],
        # target
        "goal": int(e["result"] == "SUCCESS"),
        # location
        "x": x,
        "y": y,
        "abs_y": abs(y),
        "distance": s["distance"],
        "angle_to_centre": s["angle"],
        "goal_mouth_angle": goal_mouth_angle(x, y),
        "in_box": int(x >= GOAL_X - 16.5 and abs(y) <= 20.16),
        "in_six_yard_box": int(x >= GOAL_X - 5.5 and abs(y) <= 9.16),
        # shot context
        "shot_type": e["action"],
        "body_part": e["bodyPartExtended"],
        "phase": e["phase"],
        "from_set_piece": int(e["setPiece"] is not None),
        "pressure": e["pressure"],
        "opponents": e["opponents"],
        "dist_to_opponent": DIST_TO_OPP.get(e["distanceToOpponent"], 6),
        # goalkeeper position at the moment of the shot
        "gk_known": int(gk_x is not None),
        "gk_dist_from_line": GOAL_X - gk_x if gk_x is not None else None,
        "gk_dist_to_shooter": math.hypot(gk_x - x, gk_y - y) if gk_x is not None else None,
        "gk_off_centre": abs(gk_y) if gk_y is not None else None,
        # build-up
        "prev_action": prev["actionType"] if prev and prev["squadId"] == e["squadId"] else "OTHER",
        "assist_type": assist["action"] if assist else "NONE",
        "assist_distance": assist["pass"]["distance"] if assist else None,
        "time_since_assist": e["gameTime"]["gameTimeInSec"] - assist["gameTime"]["gameTimeInSec"] if assist else None,
        # provider possession value, kept only as a benchmark (not a model input)
        "provider_pxT": e["pxT"]["team"],
    }
    row.update(set_piece_features(events, i, restarts, assist))
    return row


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "*", "*.json"))):
        league = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            events = json.load(f)
        if not isinstance(events, list):
            continue
        restarts = {}
        for e in events:
            if e.get("setPiece") and e["setPiece"]["mainEvent"]:
                restarts.setdefault(e["setPiece"]["id"], e)
        for i, e in enumerate(events):
            if e.get("shot"):
                rows.append(shot_row(events, i, league, os.path.basename(path), restarts))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} shots, {df.goal.sum()} goals -> {OUT}")
    print(df.groupby("league").goal.agg(["count", "sum", "mean"]))


if __name__ == "__main__":
    main()
