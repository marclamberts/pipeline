"""Restart / set-piece extraction.

Every event belonging to a set piece carries ``setPiece.id``; the first event
with ``mainEvent`` is the restart itself (corner, free kick, throw-in, goal
kick). Later main events in the same group are re-deliveries. We also look at
a short "second phase" window after the group ends.
"""
from __future__ import annotations

import pandas as pd

from .data import BOX_X, BOX_Y, FINAL_THIRD_X, HALF_LEN, SIX_X

SECOND_PHASE_S = 10.0   # seconds after the set piece in which the taking team's shots still count
COUNTER_S = 15.0        # seconds after the set piece in which opponent shots count as counters
LONG_THROW_M = 25.0


def _seconds(df):
    return df.t * 60


def _corner_zone(end_x, end_y, side_sign):
    if pd.isna(end_x):
        return None
    if end_x < BOX_X or abs(end_y) > BOX_Y:
        return "short / outside box"
    rel = end_y * side_sign  # > 0 = near-post side
    if end_x >= SIX_X and abs(end_y) <= 9.16:
        return "6-yard box" if abs(rel) <= 2 else ("near post" if rel > 0 else "far post")
    if end_x < HALF_LEN - 11:
        return "penalty spot / edge"
    return "near post" if rel > 2 else ("far post" if rel < -2 else "central")


def extract(df: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """One row per set piece in the match."""
    secs = _seconds(df)
    rows = []
    # Each shot belongs to at most one set piece; penalties are their own rows below.
    claimed: set[int] = set(shots.idx[shots.penalty]) if len(shots) else set()
    claimed_counter: set[int] = set()
    groups = sorted(df[df.sp_id.notna()].groupby("sp_id"), key=lambda g: g[1].index.min())
    for sp_id, grp in groups:
        mains = grp[grp.sp_main]
        main = mains.iloc[0] if len(mains) else grp.iloc[0]
        kind = main.action_type
        if kind == "SHOT":
            kind = "CORNER" if main.action == "CORNER" else "FREE_KICK"
        elif kind not in ("CORNER", "FREE_KICK", "THROW_IN", "GOAL_KICK"):
            kind = "OTHER"
        team_sid = main.squad_id
        t0, t1 = secs[grp.index.min()], secs[grp.index.max()]
        period = main.period
        in_grp = shots[shots.idx.isin(grp.index)]
        after = shots[(shots.period == period) & (shots.t * 60 > t1) & (shots.t * 60 <= t1 + SECOND_PHASE_S)
                      & ~shots.idx.isin(grp.index)]
        # Stop the second-phase window at the next restart.
        nxt = df[(df.index > grp.index.max()) & df.sp_main]
        if len(nxt):
            after = after[after.idx < nxt.index[0]]
        own_shots = pd.concat([in_grp[in_grp.squad_id == team_sid], after[after.squad_id == team_sid]])
        own_shots = own_shots[~own_shots.idx.isin(claimed)]
        claimed |= set(own_shots.idx)
        counters = shots[(shots.period == period) & (shots.squad_id != team_sid) & (shots.t * 60 > t0)
                         & (shots.t * 60 <= t1 + COUNTER_S) & ~shots.idx.isin(claimed_counter)]
        claimed_counter |= set(counters.idx)
        side = 1 if (main.y or 0) > 0 else -1  # +1 = taken from the attacking left
        foot = main.body_ext
        swing = None
        if kind == "CORNER" and foot in ("FOOT_LEFT", "FOOT_RIGHT"):
            # From the left (y>0) a right-footer curls towards goal; mirrored on the right.
            swing = "inswing" if (side > 0) == (foot == "FOOT_RIGHT") else "outswing"
        short = bool(pd.notna(main.pass_dist) and main.pass_dist < 20)
        rows.append({
            "sp_id": int(sp_id), "type": kind, "team": main.team, "squad_id": team_sid,
            "period": period, "clock": main.clock, "t": main.t,
            "taker": main.player, "foot": foot, "side": "left" if side > 0 else "right",
            "x": main.x, "y": main.y, "end_x": main.end_x, "end_y": main.end_y,
            "pass_dist": main.pass_dist, "short": short, "swing": swing,
            "direct_shot": main.action_type == "SHOT",
            "target_zone": _corner_zone(main.end_x, main.end_y, side) if kind in ("CORNER", "FREE_KICK") else None,
            "first_contact": main.receiver_type,  # TEAMMATE / OPPONENT / None
            "first_contact_player_id": main.receiver_id if main.receiver_type == "TEAMMATE" else None,
            "attacking_third": (main.x or -99) >= FINAL_THIRD_X,
            "long_throw": kind == "THROW_IN" and (main.pass_dist or 0) >= LONG_THROW_M and (main.x or -99) >= FINAL_THIRD_X - 10,
            "redeliveries": max(0, len(mains) - 1),
            "shots": int(len(own_shots)), "xg": float(own_shots.xg.sum()), "goals": int(own_shots.goal.sum()),
            "shooters": "; ".join(own_shots.player.dropna().astype(str)),
            "counter_shots_conceded": int(len(counters)), "counter_xg_conceded": float(counters.xg.sum()),
            "counter_goals_conceded": int(counters.goal.sum()),
        })
    out = pd.DataFrame(rows)
    pens = shots[shots.penalty]
    if len(pens):
        out = pd.concat([out, pd.DataFrame({
            "type": "PENALTY", "team": pens.team, "squad_id": pens.squad_id, "period": pens.period,
            "clock": pens.clock, "t": pens.t, "taker": pens.player, "x": pens.x, "y": pens.y,
            "direct_shot": True, "shots": 1, "xg": pens.xg, "goals": pens.goal.astype(int),
            "shooters": pens.player, "attacking_third": True,
        })], ignore_index=True)
    for col in ("short", "direct_shot", "attacking_third", "long_throw"):
        out[col] = out[col].astype("boolean").fillna(False).astype(bool)
    return out


def summarise(sp: pd.DataFrame, team: str, n_matches: int) -> dict:
    """Attacking and defending set-piece profile for `team` over the given set pieces."""
    att, dfn = sp[sp.team == team], sp[sp.team != team]
    per = lambda v: round(v / max(1, n_matches), 2)

    def block(s: pd.DataFrame) -> dict:
        corners = s[s.type == "CORNER"]
        fks = s[(s.type == "FREE_KICK") & s.attacking_third.fillna(False)]
        throws = s[s.type == "THROW_IN"]
        gk = s[s.type == "GOAL_KICK"]
        # Restarts that are a direct threat; a throw-in or free kick deep in our half is not.
        threat = s[s.type.isin(["CORNER", "PENALTY"])
                   | (s.type.isin(["FREE_KICK", "THROW_IN"]) & s.attacking_third)]
        delivered = corners[~corners.short]
        return {
            "corners": int(len(corners)), "corners_per_match": per(len(corners)),
            "corner_short_pct": round(100 * corners.short.mean(), 1) if len(corners) else None,
            "corner_inswing_pct": round(100 * (delivered.swing == "inswing").mean(), 1) if len(delivered) else None,
            "corner_first_contact_won_pct": round(100 * (delivered.first_contact == "TEAMMATE").mean(), 1) if len(delivered) else None,
            "corner_target_zones": delivered.target_zone.value_counts().to_dict(),
            "corner_shots": int(corners.shots.sum()), "corner_xg": round(float(corners.xg.sum()), 2),
            "corner_goals": int(corners.goals.sum()),
            "xg_per_corner": round(float(corners.xg.mean()), 3) if len(corners) else None,
            "attacking_free_kicks": int(len(fks)), "direct_fk_shots": int(fks.direct_shot.sum()),
            "fk_xg": round(float(fks.xg.sum()), 2), "fk_goals": int(fks.goals.sum()),
            "throw_ins": int(len(throws)), "long_throws": int(throws.long_throw.fillna(False).sum()),
            "long_throw_xg": round(float(throws[throws.long_throw.fillna(False)].xg.sum()), 2),
            "goal_kicks": int(len(gk)),
            "goal_kick_short_pct": round(100 * (gk.pass_dist < 30).mean(), 1) if len(gk) else None,
            "penalties": int((s.type == "PENALTY").sum()),
            "set_piece_shots": int(threat.shots.sum()), "set_piece_xg": round(float(threat.xg.sum()), 2),
            "set_piece_goals": int(threat.goals.sum()),
            "set_piece_xg_per_match": per(float(threat.xg.sum())),
            "counter_shots_conceded_after": int(threat.counter_shots_conceded.fillna(0).sum()),
            "counter_goals_conceded_after": int(threat.counter_goals_conceded.fillna(0).sum()),
        }

    takers = (att[att.type.isin(["CORNER", "FREE_KICK"]) & att.attacking_third.fillna(False)]
              .groupby(["taker", "type", "foot"], dropna=False).size().reset_index(name="n")
              .sort_values("n", ascending=False).head(8))
    shooters = (att[att.shooters.fillna("") != ""].shooters.str.split("; ").explode()
                .value_counts().head(8).rename_axis("player").reset_index(name="set_piece_shots"))
    return {
        "team": team, "matches": n_matches,
        "attacking": block(att), "defending": block(dfn),
        "takers": takers.to_dict("records"), "targets": shooters.to_dict("records"),
    }
