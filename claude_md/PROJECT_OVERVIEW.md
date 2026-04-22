# IBM AI Racing League Project Overview

## Goal

Build the fastest possible autonomous F1 lap on the TORCS Corkscrew track from a standing start.

## What Lives In This Repository

This repository contains the source code and project documentation for two parallel driving pipelines:

- A rule-based driver with automated parameter search
- A reinforcement learning pipeline based on SAC

The heavy simulator assets, training outputs, recordings, and local experiment logs are intentionally kept out of version control.

## High-Level Strategy

### Rule-Based Pipeline

- Keep `torcs_jm_par.py` as the untouched protocol reference.
- Maintain `driver.py` as the known-good baseline client.
- Use `research_driver.py` to expose the tunable configuration.
- Use `autoresearch.py` to search for faster setups.
- Use `record_best_run.py` to replay a saved config and capture full telemetry for analysis.

### RL Pipeline

- Wrap TORCS in `torcs_env.py`.
- Keep reward shaping isolated in `reward.py`.
- Train with `train_sac.py`.
- Evaluate with `eval_sac.py`.
- Record full telemetry and video with `record_best_rl_run.py`.

## Rule-Based Search Modes

`autoresearch.py` supports two operating modes:

- `global`
  Tunes one shared parameter set for the full lap.
- `documented-turns`
  Keeps a shared base config and adds local overrides for the documented Corkscrew sectors `T1` through `T11`, including `T8A`.

The sector-aware mode uses sector split times to bias future mutations toward slower parts of the lap.

## Canonical Artifacts

These files represent the current best-known state and should stay consistent:

- `torcs/gym_torcs/autoresearch_best.json`
- `torcs/gym_torcs/autoresearch_best_result.json`
- `results/best_run_records/rule_based_best_lap_summary.json`
- `results/best_run_records/rule_based_best_lap_telemetry.jsonl`
- `results/best_run_records/rule_based_best_lap_time.txt`
- `results/best_run_records/best_lap_summary.json`
- `results/best_run_records/best_lap_telemetry.jsonl`
- `results/best_run_records/best_lap_time.txt`
- `results/best_run_records/best_lap_video.mp4`

## Repository Layout

```text
IBM_RACING_LEAGUE/
|-- AGENTS.md
|-- CLAUDE.md
|-- claude_md/
|   |-- PROJECT_OVERVIEW.md
|   |-- SETUP.md
|   |-- ARCHITECTURE.md
|   |-- PHASE1_RULE_BASED.md
|   |-- PHASE2_RL.md
|   `-- TASKS.md
`-- torcs/
    `-- gym_torcs/
        |-- torcs_jm_par.py
        |-- driver.py
        |-- fastest.py
        |-- research_driver.py
        |-- autoresearch.py
        |-- track_sectors.py
        |-- record_best_run.py
        |-- rl_torcs_client.py
        |-- torcs_env.py
        |-- reward.py
        |-- train_sac.py
        |-- eval_sac.py
        `-- record_best_rl_run.py
```

## Success Criteria

- Complete full laps reliably from a standing start
- Improve lap time without increasing off-track or damage issues
- Preserve reproducible best setups and telemetry traces
- Keep RL experiments separate enough that they do not destabilize the rule-based workflow
- Maintain a public-source repository with enough documentation to replay the best result
