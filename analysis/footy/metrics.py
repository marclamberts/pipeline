"""Team and player metrics computed from a flattened match DataFrame."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import BOX_X, BOX_Y, FINAL_THIRD_X, HALF_LEN, score, side_ids

DEF_ACTIONS = ["INTERCEPTION", "GROUND_DUEL", "FOUL", "BLOCK"]
REGAINS = ["LOOSE_BALL_REGAIN", "INTERCEPTION"]
HIGH_REGAIN_X = HALF_LEN - 40.0      # within 40 m of the opponent goal
PPDA_X = -HALF_LEN + 0.6 * 2 * HALF_LEN  # opponent's own 60% of the pitch, in their frame
LANES = ["LEFT_WING", "LEFT_HALF_SPACE", "CENTER", "RIGHT_HALF_SPACE", "RIGHT_WING"]


def _in_box(x, y):
    return (x >= BOX_X) & (y.abs() <= BOX_Y)


def _goal_dist(x, y):
    return np.hypot(HALF_LEN - x, y)


def shots_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per shot with xG, outcome and the assisting pass (if any)."""
    rows = []
    shot_idx = df.index[df.action_type == "SHOT"]
    for i in shot_idx:
        s = df.loc[i]
        after = df.loc[i + 1:i + 3]
        opp_after = after[after.squad_id != s.squad_id]
        goal = s.result == "SUCCESS"
        saved = opp_after.action_type.isin(["GK_SAVE", "GK_CATCH"]).any()
        blocked = (not goal) and len(opp_after) > 0 and opp_after.iloc[0].action_type == "BLOCK"
        # Assist: last pass to the shooter in the same possession, within 6 events.
        assister, assist_type = None, None
        for j in range(i - 1, max(-1, i - 7), -1):
            p = df.loc[j]
            if p.squad_id != s.squad_id:
                break
            if p.action_type in ("PASS", "CORNER", "FREE_KICK", "THROW_IN") and p.receiver_id == s.player_id:
                assister, assist_type = p.player, p.action
                break
        rows.append({
            "idx": i, "t": s.t, "clock": s.clock, "period": s.period, "team": s.team, "squad_id": s.squad_id,
            "player": s.player, "player_id": s.player_id, "action": s.action, "body": s.body_ext,
            "phase": s.phase, "x": s.x, "y": s.y, "xg": s.xg, "goal": goal,
            "on_target": goal or saved, "blocked": blocked,
            "set_piece": pd.notna(s.sp_id) or s.phase == "SET_PIECE",
            "penalty": s.action == "PENALTY_KICK",
            "assister": assister, "assist_type": assist_type,
        })
    return pd.DataFrame(rows)


def goals_table(df: pd.DataFrame) -> list[dict]:
    squads = df.attrs["squads"]
    shots = shots_table(df)
    out = []
    for i, r in df[df.action_type.isin(["GOAL", "OWN_GOAL"])].iterrows():
        own = r.action_type == "OWN_GOAL"
        team = squads[next(s for s in squads if s != r.squad_id)] if own else r.team
        shot = shots[(shots.idx < i) & (shots.idx >= i - 3) & shots.goal]
        shot = shot.iloc[-1] if len(shot) else None
        out.append({
            "clock": r.clock, "t": round(float(r.t), 1), "team": team, "scorer": r.player, "own_goal": own,
            "type": None if shot is None else shot.action, "phase": r.phase,
            "xg": None if shot is None else round(float(shot.xg), 2),
            "assist": None if shot is None else shot.assister,
        })
    return out


def team_stats(df: pd.DataFrame, team: str, shots: pd.DataFrame | None = None) -> dict:
    us, them = side_ids(df, team)
    shots = shots_table(df) if shots is None else shots
    mine, theirs = df[df.squad_id == us], df[df.squad_id == them]
    my_shots, their_shots = shots[shots.squad_id == us], shots[shots.squad_id == them]

    passes = mine[mine.action_type == "PASS"]
    opp_passes = theirs[theirs.action_type == "PASS"]
    ok = passes.result == "SUCCESS"
    att = passes.result.isin(["SUCCESS", "FAIL"])
    moves = mine[mine.action_type.isin(["PASS", "DRIBBLE"]) & (mine.result == "SUCCESS") & mine.end_x.notna()]
    f3_entries = moves[(moves.x < FINAL_THIRD_X) & (moves.end_x >= FINAL_THIRD_X)]
    box_entries = moves[~_in_box(moves.x, moves.y) & _in_box(moves.end_x, moves.end_y)]
    prog = passes[ok & passes.end_x.notna()]
    prog = prog[(_goal_dist(prog.end_x, prog.end_y) <= 0.75 * _goal_dist(prog.x, prog.y)) & (prog.end_x > -10)]
    crosses = passes[passes.action.isin(["HIGH_CROSS", "LOW_CROSS"])]
    ppda_passes = opp_passes[opp_passes.x < PPDA_X]
    ppda_def = mine[mine.action_type.isin(DEF_ACTIONS) & (mine.x > -PPDA_X)]
    regains = mine[mine.action_type.isin(REGAINS)]
    losses = mine[mine.action_type.isin(["PASS", "DRIBBLE"]) & (mine.result == "FAIL")]
    # "Cheap" losses: failed ground passes / dribbles (not contested long balls, headers or crosses).
    cheap = losses[losses.action.isin(["LOW_PASS", "DRIBBLE"])]
    touches_f3 = (mine.x >= FINAL_THIRD_X).sum()
    opp_touches_f3 = (theirs.x >= FINAL_THIRD_X).sum()
    opp_build = opp_passes[opp_passes.x < 0]

    def r(v, n=2):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), n)

    return {
        "team": team,
        "goals": score(df)[team],
        "xg": r(my_shots.xg.sum()),
        "np_xg": r(my_shots[~my_shots.penalty].xg.sum()),
        "shots": int(len(my_shots)),
        "shots_on_target": int(my_shots.on_target.sum()),
        "big_chances": int((my_shots.xg >= 0.3).sum()),
        "xg_per_shot": r(my_shots.xg.mean()) if len(my_shots) else None,
        "set_piece_xg": r(my_shots[my_shots.set_piece].xg.sum()),
        "xg_against": r(their_shots.xg.sum()),
        "possession_pct": r(100 * len(passes) / max(1, len(passes) + len(opp_passes)), 1),
        "passes": int(len(passes)),
        "pass_accuracy_pct": r(100 * ok.sum() / max(1, att.sum()), 1),
        "long_pass_pct": r(100 * (passes.pass_dist >= 30).sum() / max(1, len(passes)), 1),
        "progressive_passes": int(len(prog)),
        "final_third_entries": int(len(f3_entries)),
        "box_entries": int(len(box_entries)),
        "crosses": int(len(crosses)),
        "cross_accuracy_pct": r(100 * (crosses.result == "SUCCESS").sum() / max(1, len(crosses)), 1),
        "field_tilt_pct": r(100 * touches_f3 / max(1, touches_f3 + opp_touches_f3), 1),
        "ppda": r(len(ppda_passes) / max(1, len(ppda_def)), 1),
        "regains": int(len(regains)),
        "high_regains": int((regains.x >= HIGH_REGAIN_X).sum()),
        "losses": int(len(losses)),
        "cheap_losses": int(len(cheap)),
        "own_half_losses": int((cheap.x < 0).sum()),
        "pressure_on_opp_buildup": r(opp_build.pressure.mean(), 1),
        "pct_opp_buildup_pressured": r(100 * (opp_build.pressure >= 50).mean(), 1) if len(opp_build) else None,
        "entries_by_lane": {l: int((f3_entries.end_lane == l).sum()) for l in LANES},
        "xg_by_phase": {k: r(v) for k, v in my_shots.groupby("phase").xg.sum().items()},
        "formation": mine.formation.mode().iat[0] if mine.formation.notna().any() else None,
        "yellow_cards": int(mine.action_type.isin(["YELLOW_CARD", "SECOND_YELLOW_CARD"]).sum()),
        "red_cards": int(mine.action_type.isin(["RED_CARD", "SECOND_YELLOW_CARD"]).sum()),
    }


def player_table(df: pd.DataFrame, team: str, shots: pd.DataFrame | None = None) -> pd.DataFrame:
    us, _ = side_ids(df, team)
    shots = shots_table(df) if shots is None else shots
    mine = df[(df.squad_id == us) & df.player_id.notna()]
    my_shots = shots[shots.squad_id == us]
    rows = []
    for pid, ev in mine.groupby("player_id"):
        passes = ev[ev.action_type == "PASS"]
        ok = passes.result == "SUCCESS"
        prog = passes[ok & passes.end_x.notna()]
        prog = prog[(_goal_dist(prog.end_x, prog.end_y) <= 0.75 * _goal_dist(prog.x, prog.y)) & (prog.end_x > -10)]
        moves = ev[ev.action_type.isin(["PASS", "DRIBBLE"]) & (ev.result == "SUCCESS") & ev.end_x.notna()]
        ps = my_shots[my_shots.player_id == pid]
        name = ev.player.iat[0]
        assisted = my_shots[my_shots.assister == name]
        rows.append({
            "player": name,
            "position": ev.position.mode().iat[0] if ev.position.notna().any() else None,
            "first_min": round(float(ev.t.min())), "last_min": round(float(ev.t.max())),
            "goals": int(ps.goal.sum()), "shots": int(len(ps)), "xg": round(float(ps.xg.sum()), 2),
            "key_passes": int(len(assisted)), "xa": round(float(assisted.xg.sum()), 2),
            "passes": int(len(passes)),
            "pass_pct": round(100 * ok.sum() / max(1, passes.result.isin(["SUCCESS", "FAIL"]).sum()), 1),
            "prog_passes": int(len(prog)),
            "box_entries": int((~_in_box(moves.x, moves.y) & _in_box(moves.end_x, moves.end_y)).sum()),
            "regains": int(ev.action_type.isin(REGAINS).sum()),
            "losses": int((ev.action_type.isin(["PASS", "DRIBBLE"]) & (ev.result == "FAIL")).sum()),
        })
    out = pd.DataFrame(rows)
    out["involvement"] = out.xg + out.xa + 0.02 * out.prog_passes + 0.02 * out.box_entries
    return out.sort_values("involvement", ascending=False).drop(columns="involvement").reset_index(drop=True)


def md_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    if df is None or len(df) == 0:
        return "_none_"
    cols = list(df.columns)
    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else format(v, floatfmt)
        return "" if v is None else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
