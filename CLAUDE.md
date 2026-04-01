# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

IBM AI Racing League — autonomous F1 race car on the **Corkscrew** track in TORCS (The Open Racing Car Simulator). Goal: fastest possible lap time from a standing start.

## Competition Context

- **Deadline**: July 1st, 2026
- **Track**: Corkscrew (TORCS built-in)
- **Car**: F1 (selected via `scr_server 1` in TORCS driver config)
- **Start**: Standing start
- **Platform**: Windows native (`wtorcs.exe`) — Python communicates via UDP (port 3001)
- **University**: AGH University of Science and Technology, Kraków
- **Required tooling**: IBM Granite must be visibly used in the project (it's part of the judging)

## Architecture

TORCS runs as a server. Our Python driver is a UDP client. Every ~20ms tick:
1. TORCS sends sensor data (speed, 19 track-edge distances, angle, trackPos, wheelSpin, rpm, gear, etc.)
2. Python computes action (steering [-1,1], accel [0,1], brake [0,1], gear [-1..6])
3. Python sends command back via UDP

Entry point: `torcs_jm_par.py` (IBM-provided starter — DO NOT modify, keep as reference). We extend from copies.

## Strategy: Hybrid (Rule-Based → RL)

**Phase 1**: Rule-based driver — tune parameters, implement corner detection, graduated braking, straight-line speed optimization. See `PHASE1_RULE_BASED.md`.

**Phase 2**: Reinforcement Learning (SAC via Stable-Baselines3) with Gym wrapper around TORCS UDP. Rule-based driver becomes fallback. See `PHASE2_RL.md`.

## Commands

```bash
# Launch TORCS (Windows) — must be running before the driver
C:\Projekty\IBM_RACING_LEAGUE\torcs\torcs\wtorcs.exe
# In GUI: Race → Quick Race → Configure Race → Corkscrew track, scr_server 1 → New Race

# Run the AI driver (separate terminal, after TORCS shows waiting screen)
cd C:\Projekty\IBM_RACING_LEAGUE\torcs\gym_torcs/
python driver.py

# Run with best-tuned params
python fastest.py

# Phase 2 RL training
pip install stable-baselines3[extra] gymnasium torch tensorboard pandas
python train_sac.py
```

## Key Files

| File | Purpose |
|------|---------|
| `claude_md/PROJECT_OVERVIEW.md` | Full competition details, links, deliverables, timeline |
| `claude_md/SETUP.md` | Installation & first run instructions |
| `claude_md/ARCHITECTURE.md` | Sensor reference, actuator commands, UDP protocol, F1 car physics notes |
| `claude_md/PHASE1_RULE_BASED.md` | Rule-based development plan with pseudocode |
| `claude_md/PHASE2_RL.md` | RL upgrade plan (SAC, reward shaping, Gym wrapper) |
| `claude_md/TASKS.md` | Working checklist and lap time log |

Source code lives in `torcs/gym_torcs/`.

## Coding Conventions

- **Language**: Python 3.x
- **No external deps in Phase 1** — only stdlib (socket, struct, math). The starter uses raw UDP.
- **Phase 2 deps**: `stable-baselines3`, `gymnasium`, `torch`, `numpy`, `pandas`, `tensorboard`
- One function per driving concept (corner detection, braking, gear shift, etc.)
- Log lap times and parameters to CSV for every test run
- Commit every improvement that beats the previous best

## F1 Car Critical Physics

- **DO NOT brake and steer simultaneously** — causes severe understeer on the F1 car
- Brake BEFORE corners (in a straight line), then release brake and turn
- Graduated braking (proportional), never binary full-brake
- Acceleration to 0 when entering corners, gentle throttle through apex
- Traction control needed — rear wheels spin easily under power

## Track Sensor Quick Reference

`track[0..18]` = 19 distance sensors from -90° to +90° in 10° steps. `track[9]` = straight ahead.
- Small `track[9]` → corner approaching
- Compare `sum(track[0:9])` vs `sum(track[10:19])` → corner direction
- All sensors short → tight section, all long → wide straight

## Important Constraints

- TORCS has a **memory leak** — restart the simulator every ~20 races during automated testing
- The Python client must respond within ~10ms or TORCS repeats the previous command
- The official submission requires a **video of the fastest lap** with university/team name visible
- The code must be in a **public GitHub repo**
- IBM Granite usage must be documented (screenshots, video, blog)
