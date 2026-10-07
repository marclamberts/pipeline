# Shot value (xG) model

Builds an expected-goals model from the event JSONs in the league folders
(Eredivisie, Bundesliga, 2. Bundesliga, 3. Liga 2026-27; all leagues are pooled for training).

```bash
pip install -r xg_model/requirements.txt
python xg_model/build_dataset.py   # -> output/shots.csv (one row per shot, all features)
python xg_model/train_xg.py        # -> models, xG per shot, tables, plots
```

## Features (all pre-shot)
- **Location:** x, |y|, distance, goal-mouth angle, angle to centre, in box / six-yard box
- **Shot:** shot type (header, 1v1, open goal, free kick, ...), body part, phase, set piece
- **Defence:** pressure, opponents goal-side, distance to nearest opponent
- **Goalkeeper:** distance from line, distance to shooter, offset from centre
- **Build-up:** previous action, assist type (cross, low pass, corner, ...), assist length, time since assist

Post-shot info (target point, keeper dive, woodwork, end location) is excluded.
Penalties get a fixed xG of 0.76. The provider's `pxT` is only used as a benchmark.

## Outputs (`output/`)
| file | content |
|---|---|
| `shots_with_xg.csv` | every shot with out-of-fold `xg_location`, `xg_full_logit`, `xg_full_gbm`, and `xg` (best model) |
| `model_scores.csv` | 5-fold, match-grouped CV metrics |
| `xg_models.joblib` | final fitted models (trained on all non-penalty shots) |
| `eredivisie_players_xg.csv`, `eredivisie_teams_xg.csv` | Eredivisie summaries |
| `calibration.png`, `xg_by_location.png`, `feature_importance.png` | diagnostics |
