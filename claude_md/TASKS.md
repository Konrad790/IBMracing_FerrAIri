# TASKS — Working Checklist

## Current Phase: Setup & Phase 1 (Rule-Based)

### Pre-flight (Do FIRST)
- [ ] Download TORCS zip from https://ibm.biz/TORCSdownloadzip
- [ ] Extract to `C:\torcs\`
- [ ] Verify `C:\torcs\torcs\wtorcs.exe` exists
- [ ] Verify `C:\torcs\gym_torcs\torcs_jm_par.py` exists
- [ ] Ensure Python 3.x is installed and on PATH
- [ ] Complete IBM Granite credential: https://ibm.biz/TORCSnewGraniteSW
- [ ] Complete AI Race League Innovation Certificate: https://ibm.biz/AIRaceLeagueCert
- [ ] Register team: https://ibm.biz/RegistrationTORCS
- [ ] Join Discord: https://discord.gg/KXhqwKqnB2

### First Run
- [ ] Launch `wtorcs.exe`
- [ ] Configure: Quick Race → Configure Race → Corkscrew track → scr_server 1
- [ ] Click New Race
- [ ] In separate terminal: `python torcs_jm_par.py`
- [ ] Verify car drives autonomously
- [ ] Record baseline lap time: ________
- [ ] Note problem areas on track: ________

### Phase 1: Rule-Based Development
- [ ] Read and annotate `torcs_jm_par.py` — understand every function
- [ ] Create backup copy of original `torcs_jm_par.py`
- [ ] Create `driver.py` (working copy)
- [ ] Implement corner detection using `track` sensors
- [ ] Implement graduated braking (not binary)
- [ ] Implement straight detection + high speed mode
- [ ] Tune F1-specific parameters (no brake+steer simultaneously)
- [ ] Implement RPM-based or speed-based gear shifting
- [ ] Add traction control for wheel spin
- [ ] Create `fastest.py` with best parameters
- [ ] Build test automation (`test_runner.py` + CSV logging)
- [ ] Run parameter sweep
- [ ] Target: reliable lap completion, fastest time: ________

### Phase 2: Reinforcement Learning
- [ ] Install dependencies: `stable-baselines3`, `gymnasium`, `torch`
- [ ] Create `torcs_env.py` — Gym wrapper around TORCS UDP communication
- [ ] Implement reward function in `reward.py`
- [ ] Create `train_sac.py` — SAC training script
- [ ] Train initial model (short episodes, basic control)
- [ ] Monitor training with TensorBoard
- [ ] Evaluate: does RL beat rule-based? Lap time: ________
- [ ] Implement hybrid driver (RL + rule-based fallback)
- [ ] Fine-tune reward shaping
- [ ] Extended training (longer episodes, full laps)
- [ ] Final RL lap time: ________

### Phase 3: Submission Prep
- [ ] Clean up code for GitHub repo
- [ ] Write README.md for repo
- [ ] Record fastest lap video (standing start, Corkscrew, university + team name visible)
- [ ] Record team introduction video (team, approach, IBM Granite usage, SkillsBuild)
- [ ] Write blog post (Medium) — optional but recommended
- [ ] Submit via https://ibm.biz/TORCSForm before July 1st

## Lap Time Log

| Date | Version | Lap Time | Notes |
|------|---------|----------|-------|
| | baseline | | First run with default params |
| | | | |
| | | | |

## Key Decisions Log

| Decision | Reasoning | Date |
|----------|-----------|------|
| Hybrid approach (rule-based → RL) | Get running fast, then optimize with RL | Apr 2026 |
| Windows native (not WSL2) | IBM provides wtorcs.exe, simpler setup | Apr 2026 |
| | | |
