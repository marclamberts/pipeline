# Football analysis toolkit

Python tooling behind the three Claude Code agents in `.claude/agents/`:

| Agent | Script | Output |
|---|---|---|
| `match-report` | `python -m analysis.match_report <team> [<team>] [--date D \| --match-id N]` | `reports/match/<match>/` |
| `opposition-report` | `python -m analysis.opposition_report <team> [--last 5] [--before D] [--venue home\|away\|all]` | `reports/opposition/<team>_<date>/` |
| `set-piece-analyst` | `python -m analysis.set_pieces <team> [--last N] [--before D] [--match-id N]` | `reports/set_pieces/<team>_<tag>/` |

Each script writes (and overwrites on re-run) `report.md` (tables, charts, and `<!-- NARRATIVE -->` placeholders the agent fills in), `stats.json` (all the numbers), and PNG charts.

Helpers:

- `python -m analysis.matches [team] [--league L] [--upcoming] [--teams]`: find matches, match ids, and team names.
- `python -m analysis.sequence --match-id N (--goals | --minute M [--window W] | --sp-id S)`: print the raw event sequence of a moment (goal build-ups, chances, set-piece routines).
- `python -m analysis.fit_xg`: rebuild `analysis/squads.json` and refit `analysis/xg_model.json` (do this after many new matches land).

Setup: `pip install -r requirements.txt` (pandas, numpy, matplotlib). The first run processes every match it needs (about 0.3 s each) and caches the results in `analysis/.cache/` (git-ignored). Later runs take a few seconds.

## Data notes

- Team names come from the filenames, because the events only carry `squadId`. Filenames lose non-ASCII letters (`M_nchen`), so `footy/data.py:NAME_FIXES` restores them for display. Team lookups are accent- and case-insensitive substring matches: "koln" is ambiguous, "1 fc koln" is not.
- Files containing `[]` are unplayed fixtures and are skipped.
- Coordinates are in metres from the centre spot (105 × 68). We use `adjCoordinates`, which are oriented so **the acting team always attacks towards +x**, with y > 0 on that team's left.
- `pressure` runs from 0 to 100 and describes the pressure on the player performing the action.

## Metric definitions

| Metric | Definition |
|---|---|
| xG | Our own logistic model on shot distance, visible goal angle, header / direct FK / open goal / 1v1 flags, fitted on every shot in the repo. Penalties = 0.76. It is **not** a provider xG. Checked against the provider model in `Eredivisie 2026-2027/Danger/`: per-shot correlation is 0.85, and ours runs about 15–20% higher on average. Treat ±0.05 per shot as noise. |
| Shots on target | Goals plus shots followed within three events by an opponent `GK_SAVE` or `GK_CATCH`. |
| Big chance | A shot with xG ≥ 0.30. |
| Possession % | Share of the match's passes. This is a proxy, not time-based possession. |
| Progressive pass | A completed pass that ends at least 25% closer to goal and in the attacking ~60% of the pitch (end x > −10). |
| Final-third entry | A completed pass or dribble that starts before and ends inside the final third (x ≥ 17.5). |
| Box entry | A completed pass or dribble that ends in the opponent box and starts outside it. |
| Field tilt % | The team's share of all touches made in the final third by either team. |
| PPDA | Opponent passes in their own 60% of the pitch ÷ our interceptions, ground duels, fouls and blocks in that zone. Lower means a more intense press. |
| Regains / high regains | `LOOSE_BALL_REGAIN` plus `INTERCEPTION`; "high" means within 40 m of the opponent goal. |
| Ground losses in own half | Failed `LOW_PASS` or `DRIBBLE` events in the team's own half. These are the "cheap" turnovers; contested long balls, headers and crosses are excluded. |
| Pressure on opp. build-up | Mean `pressure` on opponent passes in their own half. |
| Set piece | Grouped by `setPiece.id`; the type comes from the first main event. A set piece's shots are those inside the sequence or ≤10 s after it (before the next restart). Each shot counts once. **Set-piece xG / shots / goals** in the set-piece tables count only threatening restarts: corners, penalties, and free kicks or throw-ins taken in the final third. |
| Set-piece-phase xG | The team-stats row: xG of every shot the provider tags with the set-piece phase, from any restart anywhere. It is broader than the set-piece tables. |
| Corner zones | Delivery end point: 6-yard box (central), near post / far post (relative to the corner side), central, penalty spot / edge, or short / outside box. Short = pass < 20 m. |
| Inswing / outswing | Inferred from the corner side and the taker's foot. |
| First contact won | The corner's receiver is a teammate. |
| Counter shots conceded after | Opponent shots from the start of our set piece until 15 s after it ends. |
| Key pass / xA | The pass to the shooter in the same possession within 6 events, and the xG of that shot. |
| Minutes (opposition report) | Estimated from a player's first to last event in each match, so they are approximate. |
