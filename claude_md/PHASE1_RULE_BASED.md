# Phase 1: Rule-Based Driver And Autoresearch

## Goal

Produce a fast, reliable rule-based lap on Corkscrew and keep the best setup reproducible through JSON configs, saved run summaries, and telemetry logs.

## Core Files

- `torcs/gym_torcs/driver.py`
- `torcs/gym_torcs/research_driver.py`
- `torcs/gym_torcs/fastest.py`
- `torcs/gym_torcs/autoresearch.py`
- `torcs/gym_torcs/track_sectors.py`
- `torcs/gym_torcs/record_best_run.py`

## Working Loop

1. Confirm the baseline client still connects:

```powershell
python driver.py
```

2. Replay the current canonical best setup:

```powershell
python fastest.py
```

3. Run short exploratory tuning to validate the current search behavior:

```powershell
python autoresearch.py --strategy documented-turns --trials 4 --steps 8000
```

4. Run longer searches once the smoke test is stable:

```powershell
python autoresearch.py --strategy documented-turns --trials 48 --steps 8000 --exploration 0.35
```

5. Replay any saved winning config to inspect telemetry:

```powershell
python record_best_run.py --config-path ..\..\results\autoresearch\YYYYMMDD_HHMMSS\best_config.json
```

## Search Modes

### `global`

Use when:

- the baseline setup is still moving a lot
- you want broad whole-lap improvements
- sector overrides are not yet justified

### `documented-turns`

Use when:

- the global setup already completes clean laps
- bottlenecks are localized to specific corners or sections
- you want slower sectors to influence the next parameter mutations

This mode keeps the global base profile intact and stores only local override values for the documented sectors.

## Parameters That Matter Most

The most influential knobs in `DriverConfig` are usually:

- `target_speed_straight`
- `target_speed_corner`
- `steer_gain`
- `centering_gain`
- `brake_distance_threshold`
- `brake_intensity_max`
- `corner_detect_threshold`
- `turn_angle_threshold`
- `in_turn_brake_cap`
- `corner_throttle_cap`
- `tc_slip_threshold`
- `tc_throttle_cut`

The recovery and launch parameters matter too, but they usually become secondary once clean laps are already happening.

## How Sector Times Affect Tuning

In sector-aware mode, the search loop keeps sector split data for every full lap and derives `sector_times` from those splits.

That enables two useful behaviors:

- the slowest sectors can receive a higher share of future mutations
- the best replay summary can explain where time was gained or lost, instead of exposing only the final lap time

This is a refinement of the same feedback loop as before, not a separate scoring system. The primary objective is still a faster clean lap.

## Best-Artifact Rules

- `autoresearch_best.json` is the canonical best setup.
- `autoresearch_best_result.json` is the canonical metadata for that setup.
- A run-local best must not overwrite the canonical files unless the canonical best truly improved.
- Each saved run directory should be replayable on its own through `best_config.json` and `replay_best_command.txt`.

## Recommended Analysis After Each Run

Check:

- best lap time
- off-track count
- whether the best lap was clean
- sector time distribution
- how tightly clustered the top few candidates were
- whether the run improved the canonical global best or only the current run best

If the top results are tightly clustered, lower exploration and run longer. If the run is unstable, widen exploration slightly or reset to the last known-good baseline.
