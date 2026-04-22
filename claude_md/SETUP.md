# Setup And First Run Guide

## Prerequisites

- Windows with Python 3 on `PATH`
- This repository cloned to `C:\Projekty\IBM_RACING_LEAGUE`
- TORCS simulator files available under `C:\Projekty\IBM_RACING_LEAGUE\torcs\torcs`
- Optional for RL video capture: `ffmpeg` on `PATH` or a local binary passed with `--ffmpeg-path`

## Expected Local Layout

```text
C:\Projekty\IBM_RACING_LEAGUE\
|-- torcs\
|   |-- gym_torcs\        # repository code
|   `-- torcs\            # local TORCS simulator assets and wtorcs.exe
`-- tools\
    `-- ffmpeg\           # optional local ffmpeg bundle, ignored by git
```

The repository keeps source code in git, but the TORCS simulator binaries and optional ffmpeg bundle are local-only dependencies.

## Launch TORCS

1. Open a terminal in `C:\Projekty\IBM_RACING_LEAGUE\torcs\torcs`
2. Run:

```powershell
.\wtorcs.exe
```

3. In the GUI choose:
   - `Race -> Quick Race -> Configure Race`
   - track: `Corkscrew`
   - driver: `scr_server 1`
4. Start the race and wait until TORCS is on the loading or waiting screen.

## First Client Run

Open a second terminal in `C:\Projekty\IBM_RACING_LEAGUE\torcs\gym_torcs` and run:

```powershell
python driver.py
```

If the client connects correctly, the car should leave the grid and TORCS should stop waiting for input.

## Rule-Based Commands

```powershell
# Replay the current canonical best setup
python fastest.py

# Record telemetry for the canonical best setup
python record_best_run.py

# Record telemetry for a specific saved config
python record_best_run.py --config-path ..\..\results\autoresearch\YYYYMMDD_HHMMSS\best_config.json

# Run global parameter search
python autoresearch.py

# Run sector-aware parameter search
python autoresearch.py --strategy documented-turns
```

## RL Commands

Install dependencies:

```powershell
pip install -r C:\Projekty\IBM_RACING_LEAGUE\torcs\gym_torcs\rl_requirements.txt
```

Train and evaluate:

```powershell
python train_sac.py --timesteps 100000
python eval_sac.py --episodes 3
python record_best_rl_run.py --episodes 3
```

If `ffmpeg` is not on `PATH`, pass it explicitly:

```powershell
python train_sac.py --timesteps 100000 --ffmpeg-path C:\Projekty\IBM_RACING_LEAGUE\tools\ffmpeg\bin\ffmpeg.exe
python record_best_rl_run.py --episodes 3 --ffmpeg-path C:\Projekty\IBM_RACING_LEAGUE\tools\ffmpeg\bin\ffmpeg.exe
```

## Quick Troubleshooting

| Problem | Check |
|---------|-------|
| Client cannot connect | TORCS must already be running and waiting on `scr_server 1` |
| TORCS closes after the client starts | Wrong race setup or wrong driver slot |
| Autoresearch fails immediately | Launch TORCS first, then run the script from `torcs\gym_torcs` |
| RL video is missing | Install `ffmpeg` or pass `--ffmpeg-path` |
| Best config replay does not match a saved run | Use the saved `best_config.json` from that run through `--config-path` |

## Recommended First Validation

1. Confirm `python driver.py` connects and completes at least a partial lap.
2. Confirm `python fastest.py` uses the JSON-backed best setup.
3. Confirm `python record_best_run.py` writes a telemetry folder and summary.
4. Confirm `python autoresearch.py --strategy documented-turns --trials 2` completes a short smoke test before starting long runs.
