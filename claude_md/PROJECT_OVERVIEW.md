# IBM AI Racing League — Project Overview

## What Is This?

A global university competition by IBM where student teams build an autonomous AI race car in TORCS (The Open Racing Car Simulator) using Python and IBM Granite. The goal: achieve the fastest possible lap time on the **Corkscrew** track with an **F1 car**, from a **standing start**.

## Competition Details

| Field | Value |
|-------|-------|
| Organizer | IBM (AI Racing League) |
| Track | **Corkscrew** (built into TORCS) |
| Car | F1 car (via `scr_server 1` driver) |
| Start type | **Standing start** |
| Platform | Windows (native), macOS via Wine |
| Language | Python 3.x |
| AI requirement | Must use IBM Granite (code assistance, troubleshooting, optimization) |
| Deadline | **July 1st, 2026** (Mid Season Race Festival) |
| Team size | 3–5 students (currently solo, may expand) |
| University | AGH University of Science and Technology, Kraków |

## Submission Requirements

1. **Fastest Lap Video** — single fastest lap recording, university + team name visible throughout
2. **Team Introduction Video** — team intro, roles, strategic approach, how IBM Granite & SkillsBuild were used
3. **GitHub Repository** — public repo with all AI driver code
4. **Race Details** — as specified in submission form
5. **Blog (optional but recommended)** — Medium/WordPress post about the journey

**Submission portal**: https://ibm.biz/TORCSForm

## Strategy: Hybrid Approach

### Phase 1: Rule-Based AI (weeks 1–3)
- Get TORCS running with the provided `torcs_jm_par.py` starter code
- Understand sensor data and driving commands (UDP protocol)
- Implement and tune rule-based driving logic
- Target: complete laps reliably, establish baseline lap time
- Iterate: tune parameters (TARGET_SPEED, STEER_GAIN, BRAKE_THRESHOLD, etc.)

### Phase 2: Reinforcement Learning (weeks 4–8)
- Set up gym_torcs or equivalent Gym wrapper
- Implement RL agent (DDPG/SAC/PPO) for continuous control
- Use Phase 1 rule-based driver as baseline/fallback
- Train on Corkscrew specifically
- Fine-tune reward shaping for lap time minimization

### Phase 3: Polish & Submission (weeks 9–12)
- Parameter sweep automation (CSV logging, subprocess-based test runner)
- Record fastest lap video
- Prepare team video and blog
- Clean up repo for submission

## Key Links

| Resource | URL |
|----------|-----|
| TORCS Download (IBM zip) | https://ibm.biz/TORCSdownloadzip |
| Quick Start Guide | https://ibm.biz/TORCSQuickStartExt |
| IBM Granite SkillsBuild | https://ibm.biz/TORCSnewGraniteSW |
| AI Race League Registration | https://ibm.biz/RegistrationTORCS |
| Submission Form | https://ibm.biz/TORCSForm |
| Discord | https://discord.gg/KXhqwKqnB2 |
| Granite + Ollama Instructions | https://ibm.box.com/v/TorcsOllamaInstructions |
| TORCS Reference (RL, bots) | https://ibm.box.com/v/TORCSReference |
| SCR Competition Manual | https://arxiv.org/abs/1304.1672 |
| Example team repo (MonDragons) | https://github.com/Simple-wood/IBM-TORCs |

## File Structure (after IBM zip extraction)

```
C:\torcs\
├── gym_torcs\
│   └── torcs_jm_par.py      ← main AI driver script (this is what we modify)
└── torcs\
    └── wtorcs.exe            ← TORCS simulator binary (Windows)
```

## How It Works (Architecture)

```
┌─────────────┐    UDP sensors     ┌──────────────────┐
│   TORCS      │ ───────────────→  │  torcs_jm_par.py │
│  (wtorcs.exe)│ ←───────────────  │  (Python driver)  │
│              │   UDP commands     │                    │
│  scr_server 1│                   │  drive_example()   │
│  Corkscrew   │                   │  drive_modular()   │
└─────────────┘                    └──────────────────┘
```

The Python script communicates with TORCS via UDP sockets:
1. Receives sensor data every tick (speed, track position, distances, wheel spin, etc.)
2. Computes driving action (steering, throttle, brake, gear)
3. Sends commands back to TORCS
4. Loop repeats until race completion
