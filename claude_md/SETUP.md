# Setup & First Run Guide

## Prerequisites

- [x] Windows OS (10/11)
- [ ] Python 3.x installed and on PATH
- [ ] TORCS package downloaded from https://ibm.biz/TORCSdownloadzip
- [ ] IBM Granite credential completed: https://ibm.biz/TORCSnewGraniteSW
- [ ] AI Race League Innovation Certificate: https://ibm.biz/AIRaceLeagueCert

## Step 1: Extract TORCS

Download `torcs.zip` from https://ibm.biz/TORCSdownloadzip and extract to `C:\torcs\`.

Verify folder structure:
```
C:\torcs\
├── gym_torcs\
│   ├── torcs_jm_par.py      ← AI driver script
│   └── (other gym_torcs files)
└── torcs\
    ├── wtorcs.exe            ← simulator
    └── (data, tracks, cars...)
```

## Step 2: Launch TORCS Simulator

1. Navigate to `C:\torcs\torcs\`
2. Run `wtorcs.exe`
3. In the TORCS menu: **Race → Quick Race → Configure Race**

## Step 3: Configure Race

### Track Selection
- Select **Corkscrew** (this is the official competition track)

### Driver Selection
- **MUST select**: `scr_server 1` (this is the AI-controlled car slot)
- Optional: add other drivers for competition/testing

### Start Race
- Click **New Race** to proceed
- TORCS will show a loading/waiting screen (blue screen with text info)

## Step 4: Start the AI Driver

While TORCS is loading/waiting:

1. Open a **new terminal/cmd** window
2. Navigate to `C:\torcs\gym_torcs\`
3. Run:
```bash
python torcs_jm_par.py
```

The race should now start with the AI driver in control.

## Step 5: Verify It Works

You should see:
- The F1 car driving autonomously on Corkscrew
- Sensor data printing in the terminal (speed, position, etc.)
- The car completing laps (even if slowly or with issues)

Press **F2** in TORCS to cycle through camera views.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| AI car not moving | Make sure `torcs_jm_par.py` is running BEFORE or right as the race loads |
| Connection refused / timeout | Verify `scr_server 1` is selected in driver config |
| Python not found | Ensure Python 3.x is installed and `python` is on PATH |
| TORCS won't start | Check that extraction path is `C:\torcs\torcs\` (no nested extra folders) |
| Car drives but badly | Expected! Default params need tuning — see PHASE1_RULE_BASED.md |

## WSL2 Note

The official IBM setup is Windows-native (`wtorcs.exe`). The Python script communicates via UDP (localhost:3001 by default), so you CAN run the Python side in WSL2 if you prefer that dev environment. However, `wtorcs.exe` must run on Windows. For simplicity, running everything on Windows is recommended for the initial setup.

If you do use WSL2 for the Python side:
- Ensure UDP can reach between WSL2 and Windows (usually works on localhost in WSL2 with mirrored networking mode)
- TORCS GUI still runs on Windows

## First Run Checklist

```
[ ] torcs.zip downloaded and extracted to C:\torcs\
[ ] wtorcs.exe launches and shows TORCS menu
[ ] Corkscrew track selected
[ ] scr_server 1 selected as driver
[ ] New Race started (TORCS shows waiting/loading screen)
[ ] torcs_jm_par.py runs without errors
[ ] Car drives autonomously (even if poorly)
[ ] Able to see lap time in TORCS
```

## Next Steps

Once you have a working first run:
1. Read `ARCHITECTURE.md` to understand the code and sensors
2. Follow `PHASE1_RULE_BASED.md` to start optimizing
