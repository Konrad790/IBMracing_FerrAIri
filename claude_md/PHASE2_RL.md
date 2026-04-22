# Phase 2: Reinforcement Learning

## Goal

Train a SAC policy that can match or beat the best rule-based lap without destabilizing the rule-based workflow.

## Core Files

- `torcs/gym_torcs/rl_requirements.txt`
- `torcs/gym_torcs/rl_torcs_client.py`
- `torcs/gym_torcs/torcs_env.py`
- `torcs/gym_torcs/reward.py`
- `torcs/gym_torcs/train_sac.py`
- `torcs/gym_torcs/eval_sac.py`
- `torcs/gym_torcs/record_best_rl_run.py`

## Install

```powershell
pip install -r C:\Projekty\IBM_RACING_LEAGUE\torcs\gym_torcs\rl_requirements.txt
```

## Training Loop

Start TORCS first, then train from `torcs/gym_torcs`:

```powershell
python train_sac.py --timesteps 100000
```

Useful flags:

- `--max-episode-steps 9000`
- `--learning-rate`
- `--learning-rate-schedule constant|linear|cosine`
- `--learning-rate-final`
- `--buffer-size`
- `--batch-size`
- `--learning-starts`
- `--checkpoint-freq`
- `--resume-from <path>`
- `--fresh-start`
- `--device auto|cpu|cuda`
- `--ffmpeg-path <path>`
- `--no-record-video`

By default, training resumes from `models/sac_corkscrew/sac_corkscrew_latest.zip` unless `--fresh-start` is used.

## Evaluation

```powershell
python eval_sac.py --episodes 3
```

Useful flags:

- `--model <path>`
- `--max-episode-steps 9000`
- `--stochastic`
- `--device auto|cpu|cuda`

## Recording The Best RL Run

```powershell
python record_best_rl_run.py --episodes 3
```

Useful flags:

- `--model <path>`
- `--episodes`
- `--max-episode-steps`
- `--stochastic`
- `--ffmpeg-path <path>`
- `--no-record-video`
- `--tag <name>`

## Environment Summary

`torcs_env.py` currently provides:

- a 2D continuous action space
- a 32-dimensional normalized observation vector
- launch assist for the standing start
- longer stuck and off-track grace windows than a toy setup
- episode lengths suitable for full Corkscrew attempts

## Reward Summary

`reward.py` rewards:

- forward progress
- forward-aligned speed
- lap completion

and penalizes:

- center deviation
- track-angle error
- damage
- lateral motion
- off-track behavior
- backward driving
- terminal failures

## Best-Artifact Flow

Training can update:

- `results/best_run_records/rl_training_best_lap_summary.json`
- `results/best_run_records/rl_training_best_lap_telemetry.jsonl`
- `results/best_run_records/rl_training_best_lap_time.txt`
- `results/best_run_records/rl_training_best_lap_video.mp4`

If an RL lap is the best overall full lap, it can also promote:

- `results/best_run_records/best_lap_summary.json`
- `results/best_run_records/best_lap_telemetry.jsonl`
- `results/best_run_records/best_lap_time.txt`
- `results/best_run_records/best_lap_video.mp4`

## Operating Rules

- Keep the RL files and artifacts separate from the rule-based and autoresearch files.
- Do not treat a partially trained policy as the new baseline until it has verified lap-time evidence.
- Use `record_best_rl_run.py` before claiming a new RL best so the telemetry and video are saved together.
