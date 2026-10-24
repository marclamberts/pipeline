"""List matches (played by default) with optional filters.

    python -m analysis.matches                     # everything played
    python -m analysis.matches Meppen              # one team
    python -m analysis.matches --league Eredivisie --upcoming
"""
import argparse

from analysis.footy.data import find_team, match_index


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("team", nargs="?")
    ap.add_argument("--league", help="substring of the league folder name")
    ap.add_argument("--upcoming", action="store_true", help="list unplayed fixtures instead")
    ap.add_argument("--teams", action="store_true", help="list team names instead")
    args = ap.parse_args()
    ms = match_index(played_only=False)
    if args.league:
        ms = [m for m in ms if args.league.casefold() in m.league.casefold()]
    if args.teams:
        for t in sorted({m.home for m in ms} | {m.away for m in ms}):
            print(t)
        return
    if args.team:
        team = find_team(args.team)
        ms = [m for m in ms if team in (m.home, m.away)]
    ms = [m for m in ms if m.played != args.upcoming]
    for m in ms:
        print(f"{m.match_id}\t{m.date}\t{m.league}\t{m.home} vs {m.away}")
    print(f"# {len(ms)} {'upcoming' if args.upcoming else 'played'} matches")


if __name__ == "__main__":
    main()
