# AGENTS.md

## Project Goal

This repository is for the IBM AI Racing League TORCS project. The target is the fastest possible standing-start lap on the Corkscrew track with the F1 car.

## Critical Runbook

- Launch TORCS from `C:\Projekty\IBM_RACING_LEAGUE\torcs\torcs`.
- Start the simulator with `.\wtorcs.exe`, not `python .\wtorcs.exe`.
- When TORCS is waiting for the client, open a second terminal in `C:\Projekty\IBM_RACING_LEAGUE\torcs\gym_torcs`.
- Baseline working client: `python driver.py`
- Best saved tuned client: `python fastest.py`
- Best rule-based setup with telemetry recording: `python record_best_run.py`
- Global auto-tuning loop: `python autoresearch.py`
- Sector-aware auto-tuning loop: `python autoresearch.py --strategy documented-turns`
- RL dependencies: `pip install -r C:\Projekty\IBM_RACING_LEAGUE\torcs\gym_torcs\rl_requirements.txt`
- RL training loop: `python train_sac.py --timesteps 100000`
- RL evaluation: `python eval_sac.py --episodes 3`
- Best RL setup with telemetry recording: `python record_best_rl_run.py --episodes 3`

## Repository Rules

- Keep `torcs_jm_par.py` as the protocol reference and baseline starter.
- Do not break the known-good `driver.py` launch flow.
- Prefer adding new files around the current driver instead of replacing it.
- Keep the RL pipeline separate from the rule-based and autoresearch pipeline.
- Treat `results/`, `models/`, TensorBoard logs, videos, and local ffmpeg bundles as local artifacts, not repository source.

## Rule-Based And Autoresearch Notes

- `research_driver.py` contains the tunable driver, JSON schema, and sector-aware config support.
- `autoresearch_baseline.json` mirrors the original hand-tuned baseline.
- `autoresearch_best.json` is the canonical best-known rule-based setup.
- `autoresearch_best_result.json` is the canonical metadata record for that global best setup.
- Each `results/autoresearch/<timestamp>/` run should contain:
  - `configs/trial_XXX.json`
  - `best_config.json`
  - `best_result.json`
  - `replay_best_command.txt`
  - `best_telemetry.jsonl`
- Autoresearch must distinguish the run-local best from the global best.
- A slower run-local improvement must never overwrite `autoresearch_best.json` or `autoresearch_best_result.json`.
- `autoresearch.py --strategy documented-turns` should keep the current global setup as the shared base profile and add only local overrides for the documented Corkscrew sectors `T1, T2, T3, T4, T5, T6, T7, T8, T8A, T9, T10, T11`.
- Sector boundaries for the documented Corkscrew strategy are derived from `torcs/torcs/tracks/road/corkscrew/corkscrew.xml`, and the launch straight before Turn 1 is intentionally folded into `T1`.
- Trial results should preserve sector split data and derived `sector_times` so later analysis and replay can reconstruct the best lap breakdown.
- `fastest.py` should replay the canonical best rule-based setup using the original UDP client flow.
- `record_best_run.py` should:
  - accept `--config-path` for replaying any saved autoresearch config
  - preserve sector-aware configs
  - log the active sector name per step
  - save sector split times in the summary when a sector-based setup is replayed
- `record_best_run.py` should maintain:
  - `results/best_run_records/rule_based_best_lap_summary.json`
  - `results/best_run_records/rule_based_best_lap_telemetry.jsonl`
  - `results/best_run_records/rule_based_best_lap_time.txt`

## RL Notes

- `torcs_env.py` is the Gymnasium-compatible TORCS environment for SAC training.
- `reward.py` isolates reward shaping so it can be tuned without touching the client code.
- RL defaults should allow a full Corkscrew lap attempt. Keep `max_episode_steps` around `9000`, not `3000`.
- `torcs_env.py` includes a launch assist and a longer stuck grace period so the SAC policy can learn to leave the grid.
- `train_sac.py` supports `constant`, `linear`, and `cosine` learning-rate schedules.
- `train_sac.py` saves latest and best checkpoints, plus the latest replay buffer, and resumes from `models/sac_corkscrew/sac_corkscrew_latest.zip` unless `--fresh-start` is used.
- `train_sac.py` writes the best training lap artifacts to:
  - `results/best_run_records/rl_training_best_lap_summary.json`
  - `results/best_run_records/rl_training_best_lap_telemetry.jsonl`
  - `results/best_run_records/rl_training_best_lap_time.txt`
  - `results/best_run_records/rl_training_best_lap_video.mp4`
- `train_sac.py` also promotes the best full lap seen during training to the canonical:
  - `results/best_run_records/best_lap_summary.json`
  - `results/best_run_records/best_lap_telemetry.jsonl`
  - `results/best_run_records/best_lap_time.txt`
  - `results/best_run_records/best_lap_video.mp4`
- `eval_sac.py` runs saved SAC checkpoints in TORCS for lap-time verification.
- `record_best_rl_run.py` saves telemetry and optional mp4 video for evaluation runs and maintains both RL-only and canonical best lap artifacts.
- Automatic mp4 capture relies on `ffmpeg` being available on `PATH` or passed explicitly with `--ffmpeg-path`.
