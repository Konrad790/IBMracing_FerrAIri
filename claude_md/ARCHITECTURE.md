# Architecture And Runtime Flow

## Runtime Topology

TORCS runs as the UDP server. The Python side is a client that receives sensor packets, computes control outputs, and sends actions back on each simulation tick.

```text
wtorcs.exe  <->  UDP client in torcs/gym_torcs
```

The known-good protocol reference is `torcs_jm_par.py`. Everything else in this repository builds around that baseline.

## Rule-Based Path

### Core Components

- `driver.py`
  Baseline working client used for smoke tests.
- `research_driver.py`
  Houses the tunable rule-based driver and JSON schema.
- `fastest.py`
  Loads the canonical best setup and runs it through the original UDP loop.
- `record_best_run.py`
  Replays a chosen config, records per-step telemetry, and promotes improved canonical best-lap artifacts.

### Configuration Model

`research_driver.py` defines:

- `DriverConfig`
  Global parameter set for steering, braking, target speeds, traction control, and recovery behavior
- `DriverSetup`
  Wrapper around `DriverConfig` that optionally adds a sector layout plus per-sector overrides

`DriverSetup` supports:

- full-lap tuning with one shared config
- sector-aware tuning with a shared base config plus local overrides

## Sector-Aware Autoresearch

### Sector Model

`track_sectors.py` defines the documented Corkscrew sectors:

- `T1`
- `T2`
- `T3`
- `T4`
- `T5`
- `T6`
- `T7`
- `T8`
- `T8A`
- `T9`
- `T10`
- `T11`

The sector mapping is derived from `distFromStart` and normalized against track length. The launch straight before Turn 1 is intentionally part of `T1`.

### Search Loop

`autoresearch.py` works like this:

1. Load the canonical best setup or the baseline if `--reset-best` is used.
2. Evaluate the starting point.
3. Mutate parameters for a series of candidate trials.
4. Keep separate notions of:
   - run-local best
   - canonical global best
5. Persist the per-run best artifacts under `results/autoresearch/<timestamp>/`.
6. Update `autoresearch_best.json` and `autoresearch_best_result.json` only if the canonical global best truly improved.

### Feedback Signals

Each trial tracks:

- lap time
- off-track count
- damage
- mean speed
- termination reason
- sector splits
- derived `sector_times`

In `documented-turns` mode, slower sectors receive more mutation attention in future candidates.

## Telemetry And Best-Lap Artifacts

### Rule-Based

Per-run autoresearch outputs:

- `results/autoresearch/<timestamp>/summary.json`
- `results/autoresearch/<timestamp>/best_config.json`
- `results/autoresearch/<timestamp>/best_result.json`
- `results/autoresearch/<timestamp>/best_telemetry.jsonl`
- `results/autoresearch/<timestamp>/replay_best_command.txt`

Canonical rule-based best replay outputs:

- `results/best_run_records/rule_based_best_lap_summary.json`
- `results/best_run_records/rule_based_best_lap_telemetry.jsonl`
- `results/best_run_records/rule_based_best_lap_time.txt`

### RL

Training and evaluation can promote:

- `results/best_run_records/rl_training_best_lap_summary.json`
- `results/best_run_records/rl_training_best_lap_telemetry.jsonl`
- `results/best_run_records/rl_training_best_lap_time.txt`
- `results/best_run_records/rl_training_best_lap_video.mp4`

Canonical cross-pipeline best lap artifacts:

- `results/best_run_records/best_lap_summary.json`
- `results/best_run_records/best_lap_telemetry.jsonl`
- `results/best_run_records/best_lap_time.txt`
- `results/best_run_records/best_lap_video.mp4`

## RL Path

### Core Components

- `rl_torcs_client.py`
  RL-oriented TORCS client wrapper
- `torcs_env.py`
  Gymnasium environment
- `reward.py`
  Reward breakdown and shaping
- `train_sac.py`
  SAC training entry point, checkpointing, and best-lap promotion
- `eval_sac.py`
  Saved-model evaluation
- `record_best_rl_run.py`
  Full telemetry and optional video capture for the best RL checkpoint

### Environment Shape

`torcs_env.py` currently uses:

- a 2D continuous action space
- a 32-value normalized observation vector
- long episode limits suitable for a full Corkscrew lap
- launch assist and generous stuck handling so the policy can learn to leave the grid

### Reward Design

`reward.py` combines:

- forward progress
- forward speed
- center deviation penalty
- angle penalty
- damage penalty
- lateral motion penalty
- off-track penalty
- backward penalty
- lap completion bonus
- terminal penalty

The reward logic is deliberately isolated so it can be tuned without modifying the networking layer.
