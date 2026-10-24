"""Print the raw event sequence of a match around a moment, for explaining goals,
chances and set-piece routines.

    python -m analysis.sequence --match-id 268382 --minute 20.3          # ±0.5 min around 20:18
    python -m analysis.sequence --match-id 268382 --minute 20 --window 1.5
    python -m analysis.sequence --match-id 268382 --goals                 # every goal's build-up
    python -m analysis.sequence --match-id 268382 --sp-id 90764715        # one set piece
"""
import argparse

import pandas as pd

from analysis.footy.cache import bundle
from analysis.footy.data import find_match

COLS = ["clock", "team", "player", "position", "action_type", "action", "result", "phase",
        "x", "y", "end_x", "end_y", "pressure", "xg"]


def show(ev: pd.DataFrame):
    with pd.option_context("display.max_rows", 500, "display.width", 250, "display.max_columns", 30):
        print(ev[COLS].round(1).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-id", type=int, required=True)
    ap.add_argument("--minute", type=float, help="match minute (stoppage time continues the count, e.g. 47.5)")
    ap.add_argument("--window", type=float, default=0.5, help="minutes before/after --minute")
    ap.add_argument("--goals", action="store_true", help="the 25 events before each goal")
    ap.add_argument("--sp-id", type=int, help="all events of one set piece (sp_id from set_pieces.csv/stats)")
    args = ap.parse_args()
    m = find_match(match_id=args.match_id)
    df, _, _ = bundle(m)
    print(f"# {m.label}  (coordinates: acting team attacks →, metres from centre; y>0 = its left)")
    if args.goals:
        for i in df.index[df.action_type.isin(["GOAL", "OWN_GOAL"])]:
            print(f"\n## {df.at[i, 'action_type']} {df.at[i, 'clock']} {df.at[i, 'team']}")
            show(df.loc[max(0, i - 25):i])
    elif args.sp_id:
        ev = df[df.sp_id == args.sp_id]
        show(df.loc[ev.index.min():ev.index.max() + 8])
    elif args.minute is not None:
        show(df[(df.t >= args.minute - args.window) & (df.t <= args.minute + args.window)])
    else:
        ap.error("give --minute, --goals or --sp-id")


if __name__ == "__main__":
    main()
