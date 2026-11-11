# Transition Anatomy Model

This is a descriptive model of attacking transitions, built on the Impect event data in `3. Liga 2026-2027/`.

It deliberately avoids two things:
- **xT, pxT or any other value-surface model.**
- **Fitted or predictive probabilities.**

Every output is something observed on the pitch: a count, a time, a distance, a share, or an ordinal stage the transition actually reached.

```bash
python3 transitions/transition_model.py                       # 3. Liga (default)
python3 transitions/transition_model.py "2. Bundesliga 2026-2027"
```

It needs only `pandas` and `numpy`. Outputs are written to `transitions/output/` (see `REPORT.md` there for the current tables).

## What counts as a transition

A transition is a maximal run of consecutive events where `phase == ATTACKING_TRANSITION`, for one squad, within one period. Impect tags these phases itself, so we aren't inventing a definition.

If the team won the ball during a contested `SECOND_BALL` phase just before the run, the regain point is moved back to that event. Otherwise many regains would look like they started with a pass.

## The nine descriptors

| # | Layer | Fields | How it is measured |
|---|---|---|---|
| 1 | Trigger | `trigger` | How the ball was won: interception, ground duel (tackle), loose ball, second-ball win, GK catch, misplaced opponent pass |
| 2 | Regain zone | `regain_x/y`, `regain_third`, `regain_lane` | `adjCoordinates`, where every team attacks towards +x |
| 3 | Context | `pressure_at_regain`, `opp_goal_side_at_regain` | Impect pressure on the regainer (0–100), and the number of the losing team's players goal-side (rest defence) |
| 4 | Tempo | `duration_s`, `t_to_final_third_s`, `t_to_box_s`, `vertical_speed_ms` | Clock time between events. Speed = forward metres ÷ seconds to the furthest point |
| 5 | Directness | `directness` | Forward metres gained ÷ metres the ball actually travelled (1 = straight at goal) |
| 6 | Packing | `opponents_bypassed` | Impect `opponents` (opponents goal-side) at the regain minus the fewest reached during the run |
| 7 | Reach ladder | `reach` 0–5 | The furthest stage reached: own half → middle → final third → box → shot → goal |
| 8 | Ending | `ending` | `SHOT`, `SETTLED` (became normal possession), `LOST`, `OUT_OF_PLAY`, `FOULED`, `WON_SET_PIECE` or `CONTESTED_KEPT` |
| 9 | Archetype | `archetype` | Transparent rules applied to the fields above (see below) |

### Archetype rules

The rules are evaluated in order, and the first match wins:

1. Regain in the attacking third: **High regain strike** if it reached the box, otherwise **High regain recycle**.
2. Lost to the opponent within 5 s: **Immediate loss**.
3. First pass of 30 m or more, and 20 m or more of progress: **Long-ball release**.
4. Reached the box at 4 m/s or faster: **Direct counter** with 3 or fewer passes, otherwise **Combined counter**.
5. Reached the box, but slower: **Slow build to box**.
6. Became settled possession with less than 15 m of progress: **Secure & reset**.
7. Gained 20 m or more, then stopped: **Progress then stalled**.
8. Anything else: **Stalled early**.

The thresholds are coaching choices, not fitted values. Change them in `describe_run()`.

### The defensive mirror

The same transitions are re-read from the side of the team that lost the ball:

- **Counter-press kill rate:** the share of the opponent's transitions that the team ended by regaining the ball within 5 s.
- **Tactical-foul rate:** the share of the opponent's transitions that the team stopped with a foul.
- **Rest defence:** the team's players goal-side when it loses the ball in its attacking half.
- **Opponent → box / → shot:** what the team concedes in transition.

### Player layer

Between each pair of consecutive actions a team makes in a transition, the actor is credited with:
- the drop in opponents goal-side (opponents bypassed), and
- the forward metres gained.

Regains each player started, and how many of those reached the box, are counted alongside.

## Other approaches we could use

These all avoid both xT and predictive models:

1. **Packing / opponents bypassed:** already used (layer 6). It comes directly from Impect's `opponents` field.
2. **Tempo and verticality:** time to the final third and box, and metres per second (layer 4). This separates "fast" teams from "direct" teams.
3. **Empirical outcome ladders:** plain conditional frequencies by zone, trigger and packing band (`ladder_by_*.csv`). These are counts, not a model.
4. **Rule-based archetypes:** coach-readable categories (layer 9) instead of clustering or ML.
5. **Defensive response:** counter-press kills, tactical fouls and rest defence (the mirror above).
6. **Sequence grammar / n-grams:** count the action strings (e.g. `INTERCEPTION → DRIBBLE:PROGRESSIVE_CARRY → PASS → SHOT`). This finds each team's recurring patterns. A natural next step.
7. **Pass networks restricted to transitions:** who links to whom in the first 3 actions after a regain. This shows outlet players and release points.
8. **Lane-switch and width usage:** `lanes_used`, plus the side the ball is released to compared with the side it was won on.
9. **Game-state splits:** the same profiles split by score state and minute bands, to see whether teams counter more when leading.
10. **Opposition-specific matchups:** each team's transition profile against the defensive profile of the opponent it faces next (scouting reports).

## Transition pass networks

```bash
python3 transitions/pass_networks.py
```

This builds pass networks from transition passes only. It uses the same transitions as the model above.

- **Nodes:** each player's average location while acting in transitions. Size = number of actions.
- **Edges:** completed passes from passer to receiver. Line weight = number of passes; the arrow shows direction.
- **Outlet view:** only the first completed pass after each regain (regainer → outlet player). This shows who each team looks for first.
- **Team summary:**
  - completed passes per game and per transition
  - completion rate
  - share of passes gaining 5 m or more
  - main outlet, and the share of first passes going to the top one and top two outlets
  - the hub (the player involved in the most passes)
  - the strongest passing link

Outputs:
- `output/pass_network_nodes.csv`
- `output/pass_network_edges.csv`
- `output/pass_network_teams.csv`
- `output/pass_networks.html`: an interactive page you can choose a team on.

The page shows each team's 14 most-involved players. Players that sit too close together are spread apart for display only; their real average positions are in the tables.
