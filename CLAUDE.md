# CLAUDE.md

This file provides working guidance for coding agents and contributors in this repository.

## Project

IBM AI Racing League TORCS project. The objective is the fastest possible standing-start lap on the Corkscrew track with the F1 car.

## Current Workflow

1. Keep `torcs_jm_par.py` as the untouched protocol reference.
2. Use `driver.py` as the baseline working UDP client.
3. Use `research_driver.py` plus JSON configs for tunable rule-based driving.
4. Use `autoresearch.py` for automated search:
   - `global` tunes one setup for the whole lap
   - `documented-turns` keeps a shared base setup and mutates sector-local overrides
5. Replay and record the best rule-based setup with `record_best_run.py`.
6. Train SAC separately through `torcs_env.py`, `reward.py`, `train_sac.py`, `eval_sac.py`, and `record_best_rl_run.py`.

## Commands

```powershell
# Launch TORCS first
cd C:\Projekty\IBM_RACING_LEAGUE\torcs\torcs
.\wtorcs.exe

# In a second terminal, after TORCS is waiting for the client
cd C:\Projekty\IBM_RACING_LEAGUE\torcs\gym_torcs
python driver.py

# Replay the current best rule-based setup
python fastest.py

# Record the current best rule-based lap
python record_best_run.py

# Run global parameter search
python autoresearch.py

# Run sector-aware parameter search
python autoresearch.py --strategy documented-turns

# Install RL dependencies
pip install -r C:\Projekty\IBM_RACING_LEAGUE\torcs\gym_torcs\rl_requirements.txt

# Train and evaluate SAC
python train_sac.py --timesteps 100000
python eval_sac.py --episodes 3

# Record the best RL run
python record_best_rl_run.py --episodes 3
```

## Important Files

| File | Purpose |
|------|---------|
| `AGENTS.md` | Repository runbook, artifact rules, and pipeline constraints |
| `claude_md/PROJECT_OVERVIEW.md` | High-level project summary and repository layout |
| `claude_md/SETUP.md` | Clone, install, and first-run instructions |
| `claude_md/ARCHITECTURE.md` | Runtime topology and artifact flow |
| `claude_md/PHASE1_RULE_BASED.md` | Rule-based and autoresearch workflow |
| `claude_md/PHASE2_RL.md` | SAC workflow, artifacts, and recording flow |
| `claude_md/TASKS.md` | Current checklist and ongoing work items |
| `torcs/gym_torcs/research_driver.py` | Tunable rule-based driver and JSON schema |
| `torcs/gym_torcs/autoresearch.py` | Automated parameter search |
| `torcs/gym_torcs/track_sectors.py` | Documented Corkscrew sector boundaries |
| `torcs/gym_torcs/record_best_run.py` | Rule-based telemetry recording and best-lap promotion |
| `torcs/gym_torcs/torcs_env.py` | Gymnasium-compatible TORCS environment |
| `torcs/gym_torcs/train_sac.py` | SAC training entry point |
| `torcs/gym_torcs/record_best_rl_run.py` | RL telemetry and video recording |

## Best-Artifact Conventions

- Canonical rule-based setup: `torcs/gym_torcs/autoresearch_best.json`
- Canonical rule-based result metadata: `torcs/gym_torcs/autoresearch_best_result.json`
- Rule-based best replay artifacts: `results/best_run_records/rule_based_best_lap_*`
- Canonical cross-pipeline best lap artifacts: `results/best_run_records/best_lap_*`
- Per-run autoresearch outputs live under `results/autoresearch/<timestamp>/`

## Notes

- `results/`, `models/`, simulator binaries, TensorBoard logs, videos, and local ffmpeg bundles are local runtime artifacts and should not be committed.
- `documented-turns` tuning is driven by `distFromStart` sectors and sector times, not by a separate standalone driver.
- A run-local best result may improve the current run without replacing the canonical global best. Keep those concepts separate in docs and code.
