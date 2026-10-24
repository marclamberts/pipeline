"""Per-match disk cache of the processed frames (events with xG, shots, set pieces).

Keyed on the match file's mtime and the xG model file, so it refreshes itself
when either changes. Lives in analysis/.cache (git-ignored).
"""
from __future__ import annotations

import pickle

from . import setpieces
from .data import REPO, Match, load_events
from .metrics import shots_table
from .xg import MODEL_PATH, add_xg

CACHE_DIR = REPO / "analysis" / ".cache"
VERSION = 1  # bump when the processing code changes


def bundle(match: Match):
    """(events, shots, set_pieces) for one match."""
    key = (VERSION, match.path.stat().st_mtime_ns, MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.exists() else 0)
    path = CACHE_DIR / f"{match.match_id}.pkl"
    if path.exists():
        try:
            cached = pickle.loads(path.read_bytes())
            if cached["key"] == key:
                return cached["df"], cached["shots"], cached["sp"]
        except Exception:
            pass
    df = add_xg(load_events(match))
    shots = shots_table(df)
    sp = setpieces.extract(df, shots)
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_bytes(pickle.dumps({"key": key, "df": df, "shots": shots, "sp": sp}))
    return df, shots, sp
