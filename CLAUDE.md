# pipeline

Football event data for the 2026-27 season (Bundesliga, 2. Bundesliga, 3. Liga, Eredivisie). There is one JSON file per match, named `YYYY-MM-DD_Home_vs_Away_<matchId>.json`; unplayed fixtures are `[]`. `Eredivisie 2026-2027/Danger/` holds a few provider shot-model CSVs.

- Analysis code lives in `analysis/`. Metric definitions and data notes are in `analysis/README.md`.
- Agents live in `.claude/agents/`: `match-report`, `opposition-report`, `set-piece-analyst`. Their outputs go to `reports/`.
- Install dependencies with `pip install -r requirements.txt`. Run modules from the repo root (`python -m analysis.<module>`).
- Don't hand-parse the raw JSON (files are ~3.5 MB each). Use `analysis.footy.cache.bundle(match)` to get events, shots and set pieces.
