"""Loading and flattening of the per-match event JSON files.

Layout: one folder per league/season ("Bundesliga 2026-2027/"), one file per
match named ``YYYY-MM-DD_Home_Team_vs_Away_Team_<matchId>.json``. Unplayed
fixtures are stored as ``[]``.

Events carry a ``squadId`` but no team name, so names are recovered from the
filenames: a squad id that appears in several matches belongs to the only team
name common to all of them. The result is cached in ``analysis/squads.json``.
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SQUAD_CACHE = REPO / "analysis" / "squads.json"
FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(.+)_vs_(.+)_(\d+)\.json$")

# Pitch geometry (metres). adjCoordinates are oriented so the acting team
# always attacks towards +x; y > 0 is the attacking team's left.
HALF_LEN, HALF_WID = 52.5, 34.0
BOX_X, BOX_Y = HALF_LEN - 16.5, 20.16
SIX_X = HALF_LEN - 5.5
POST_Y = 3.66
FINAL_THIRD_X = HALF_LEN - 35.0


@dataclass(frozen=True)
class Match:
    path: Path
    league: str
    date: str
    home: str
    away: str
    match_id: int

    @property
    def played(self) -> bool:
        return self.path.stat().st_size > 10

    @property
    def label(self) -> str:
        return f"{self.date} {self.home} vs {self.away}"

    @property
    def slug(self) -> str:
        return self.path.stem


# Filenames replace non-ASCII letters with "_"; restore them for display.
NAME_FIXES = [
    ("M_nchengladbach", "Mönchengladbach"), ("M_nchen", "München"), ("M_nster", "Münster"),
    ("K_ln", "Köln"), ("N_rnberg", "Nürnberg"), ("Saarbr_cken", "Saarbrücken"), ("D_sseldorf", "Düsseldorf"),
    ("Preu_en", "Preußen"), ("Gro_aspach", "Großaspach"), ("F_rth", "Fürth"), ("Osnabr_ck", "Osnabrück"),
    ("W_rzburger", "Würzburger"),
]


def pretty(name: str) -> str:
    for broken, fixed in NAME_FIXES:
        name = name.replace(broken, fixed)
    return name.replace("_", " ")


def _fold(s: str) -> str:
    s = s.casefold().replace("_", " ").replace("ß", "ss")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def match_index(played_only: bool = True) -> list[Match]:
    matches = []
    for path in sorted(REPO.glob("*/*.json")):
        m = FILE_RE.match(path.name)
        if not m:
            continue
        date, home, away, mid = m.groups()
        match = Match(path, path.parent.name, date, pretty(home), pretty(away), int(mid))
        if not played_only or match.played:
            matches.append(match)
    return sorted(matches, key=lambda m: (m.date, m.match_id))


def teams(matches: list[Match] | None = None) -> list[str]:
    matches = matches if matches is not None else match_index(played_only=False)
    return sorted({m.home for m in matches} | {m.away for m in matches})


def team_candidates(query: str) -> list[str]:
    names = teams()
    q = _fold(query)
    exact = [n for n in names if _fold(n) == q]
    return exact or [n for n in names if q in _fold(n)]


def find_team(query: str) -> str:
    """Resolve a loose team name ("meppen", "bayern") to its filename form."""
    names = teams()
    subs = team_candidates(query)
    if len(subs) == 1:
        return subs[0]
    if len(subs) > 1:
        # "Stuttgart" -> "VfB Stuttgart" rather than "VfB Stuttgart II".
        base = min(subs, key=len)
        if all(n.startswith(base) for n in subs):
            return base
        raise SystemExit(f"'{query}' is ambiguous: {', '.join(subs)}")
    close = difflib.get_close_matches(query, names, n=5, cutoff=0.5)
    raise SystemExit(f"No team matches '{query}'." + (f" Did you mean: {', '.join(close)}?" if close else ""))


def find_match(*team_queries: str, date: str | None = None, match_id: int | None = None) -> Match:
    played = match_index()
    if match_id is not None:
        for m in match_index(played_only=False):
            if m.match_id == match_id:
                if not m.played:
                    raise SystemExit(f"{m.label} has no events yet (unplayed fixture).")
                return m
        raise SystemExit(f"No match with id {match_id}.")
    options = [team_candidates(q) or [find_team(q)] for q in team_queries]
    cands = [m for m in played if all(any(n in (m.home, m.away) for n in opts) for opts in options)]
    names = [find_team(q) for q in team_queries] if not cands else team_queries
    if date:
        cands = [m for m in cands if m.date == date]
    if not cands:
        raise SystemExit(f"No played match found for {names} {date or ''}".strip())
    return cands[-1]


def team_matches(team: str, last: int | None = None, before: str | None = None,
                 venue: str = "all") -> list[Match]:
    ms = [m for m in match_index() if team in (m.home, m.away)]
    if before:
        ms = [m for m in ms if m.date < before]
    if venue == "home":
        ms = [m for m in ms if m.home == team]
    elif venue == "away":
        ms = [m for m in ms if m.away == team]
    return ms[-last:] if last else ms


# --------------------------------------------------------------------------- squads

def _squad_ids(path: Path) -> list[int]:
    return sorted({int(s) for s in re.findall(r'"squadId":\s*(\d+)', path.read_text())})


def squad_names(refresh: bool = False) -> dict[int, str]:
    cache = {} if refresh or not SQUAD_CACHE.exists() else json.loads(SQUAD_CACHE.read_text())
    seen = cache.get("matches", {})
    dirty = False
    for m in match_index():
        if str(m.match_id) not in seen:
            seen[str(m.match_id)] = {"teams": [m.home, m.away], "squads": _squad_ids(m.path)}
            dirty = True
    candidates: dict[int, set[str]] = {}
    for rec in seen.values():
        for sid in rec["squads"]:
            candidates[sid] = candidates.get(sid, set(rec["teams"])) & set(rec["teams"])
    # Resolve leftovers by elimination within each match (a team seen once).
    names = {sid: next(iter(c)) for sid, c in candidates.items() if len(c) == 1}
    for rec in seen.values():
        unknown = [s for s in rec["squads"] if s not in names]
        known = {names[s] for s in rec["squads"] if s in names}
        left = [t for t in rec["teams"] if t not in known]
        if len(unknown) == 1 and len(left) == 1:
            names[unknown[0]] = left[0]
    if dirty or refresh:
        SQUAD_CACHE.write_text(json.dumps(
            {"names": {str(k): v for k, v in sorted(names.items())}, "matches": seen},
            indent=1, ensure_ascii=False))
    return names


# --------------------------------------------------------------------------- events

def _clock(gt: str) -> float:
    m = re.match(r"(\d+):(\d+(?:\.\d+)?)(?:\s*\(\+(\d+):(\d+(?:\.\d+)?)\))?", gt or "")
    if not m:
        return np.nan
    base = int(m.group(1)) + float(m.group(2)) / 60
    if m.group(3):
        base += int(m.group(3)) + float(m.group(4)) / 60
    return base


def _g(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k)
    return d


def load_events(match: Match) -> pd.DataFrame:
    """Flatten one match into a DataFrame (one row per event)."""
    raw = json.loads(match.path.read_text())
    names = squad_names()
    sids = sorted({e["squadId"] for e in raw if e["squadId"] is not None})
    team_of = {s: names.get(s) for s in sids}
    # Fallback when a squad cannot be resolved (team with a single match and no partner info).
    for s in sids:
        if team_of[s] is None:
            others = {v for v in team_of.values() if v}
            left = [t for t in (match.home, match.away) if t not in others]
            team_of[s] = left[0] if left else f"Squad {s}"

    rows = []
    for e in raw:
        st, en = e.get("start") or {}, e.get("end") or {}
        rows.append({
            "idx": e["index"],
            "period": e["periodId"],
            "clock": _g(e, "gameTime", "gameTime"),
            "minute": _clock(_g(e, "gameTime", "gameTime")),
            "squad_id": e["squadId"],
            "team": team_of.get(e["squadId"]),
            "att_squad_id": e.get("currentAttackingSquadId"),
            "player_id": _g(e, "player", "id"),
            "player": _g(e, "player", "name"),
            "position": _g(e, "player", "position"),
            "action_type": e["actionType"],
            "action": e["action"],
            "phase": e["phase"],
            "result": e["result"],
            "x": _g(st, "adjCoordinates", "x"), "y": _g(st, "adjCoordinates", "y"),
            "end_x": _g(en, "adjCoordinates", "x"), "end_y": _g(en, "adjCoordinates", "y"),
            "zone": st.get("packingZone"), "third": st.get("pitchPosition"), "lane": st.get("lane"),
            "end_zone": en.get("packingZone"), "end_third": en.get("pitchPosition"), "end_lane": en.get("lane"),
            "pressure": e.get("pressure"),
            "pressing_player_id": e.get("pressingPlayerId"),
            "pxt_team": _g(e, "pxT", "team"), "pxt_opp": _g(e, "pxT", "opponent"),
            "pass_dist": _g(e, "pass", "distance"),
            "receiver_id": _g(e, "pass", "receiver", "playerId"),
            "receiver_type": _g(e, "pass", "receiver", "type"),
            "body_part": e.get("bodyPart"), "body_ext": e.get("bodyPartExtended"),
            "shot_dist": _g(e, "shot", "distance"),
            "target_y": _g(e, "shot", "targetPoint", "y"), "target_z": _g(e, "shot", "targetPoint", "z"),
            "dribble_type": _g(e, "dribble", "type"),
            "duel_type": _g(e, "duel", "duelType"),
            "sp_id": _g(e, "setPiece", "id"),
            "sp_main": bool(_g(e, "setPiece", "mainEvent")),
            "duration": e.get("duration") or 0.0,
            "formation": _g(e, "formation", "team"),
            "seq": e.get("sequenceIndex"),
        })
    df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    # Continuous match minute for plotting (first-half stoppage pushes the 2nd half along).
    p1_end = df.loc[df.period == 1, "minute"].max()
    shift = max(0.0, (p1_end if pd.notna(p1_end) else 45.0) - 45.0)
    df["t"] = np.where(df.period == 2, df.minute + shift, df.minute)
    df.attrs.update(match=match, squads=team_of)
    return df


def side_ids(df: pd.DataFrame, team: str) -> tuple[int, int]:
    """(squad id of `team`, squad id of its opponent) in this match."""
    inv = {v: k for k, v in df.attrs["squads"].items()}
    us = inv[team]
    them = next(s for s in df.attrs["squads"] if s != us)
    return us, them


def score(df: pd.DataFrame) -> dict[str, int]:
    squads = df.attrs["squads"]
    goals = {name: 0 for name in squads.values()}
    for _, r in df[df.action_type.isin(["GOAL", "OWN_GOAL"])].iterrows():
        sid = r.squad_id if r.action_type == "GOAL" else next(s for s in squads if s != r.squad_id)
        goals[squads[sid]] += 1
    return goals
