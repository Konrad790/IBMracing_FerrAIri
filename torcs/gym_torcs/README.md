# IBM Racing League: Expert Best-Lap Driver

This branch contains the rule-based expert driver and the tuning setup that produced the current best saved Corkscrew lap.

Best saved lap:

```text
101.548 seconds
```

The target is a standing-start lap on the Corkscrew track with the F1 car.

## Method

The driver is not a neural network. It is an expert-style controller built from racing rules:

- read TORCS sensors for track angle, car position, speed, RPM, wheel spin, and track distances
- compute a target speed from the road distance ahead
- steer using track angle plus center-line correction
- brake before corners and limit braking while turning
- shift gears from RPM thresholds
- reduce throttle when rear-wheel slip is too high

The first stable version was moved into `research_driver.py` as a tunable driver. The parameters were then optimized by the tuning loop.

The final improvement was sector-aware tuning. The Corkscrew track is split into documented sectors `T1` through `T11` in `track_sectors.py`. The loop starts from the current best setup, mutates selected parameters, runs TORCS, compares the result, and keeps the candidate only when it improves the best known lap. Slow or failing sectors are selected more often for mutation.

## Files

```text
autoresearch.py                 tuning loop
research_driver.py              expert driver and tunable parameters
track_sectors.py                Corkscrew sector definitions
autoresearch_baseline.json      original baseline parameters
autoresearch_best.json          best saved parameters
autoresearch_best_result.json   metadata for the best saved lap
fastest.py                      replay the best saved parameters
record_best_run.py              replay and record telemetry/summary
```

## Run Best Lap

Start TORCS first:

```powershell
cd C:\Projekty\IBM_RACING_LEAGUE\torcs\torcs
.\wtorcs.exe
```

When TORCS is waiting for the client, run:

```powershell
cd C:\Projekty\IBM_RACING_LEAGUE\torcs\gym_torcs
python fastest.py
```

`fastest.py` only replays `autoresearch_best.json`. It does not overwrite the best result.

## Continue Tuning

Start TORCS as above, then run:

```powershell
cd C:\Projekty\IBM_RACING_LEAGUE\torcs\gym_torcs
python autoresearch.py --strategy documented-turns
```

This can update `autoresearch_best.json` and `autoresearch_best_result.json` if a faster lap is found.

## Record Telemetry

```powershell
cd C:\Projekty\IBM_RACING_LEAGUE\torcs\gym_torcs
python record_best_run.py
```

Telemetry and summaries are local artifacts and should not be committed.
