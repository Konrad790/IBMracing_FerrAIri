# Architecture & Sensor Reference

## Communication Protocol

TORCS uses the **SCR (Simulated Car Racing) protocol** — a UDP-based client-server architecture:

- **Server**: TORCS simulator (`wtorcs.exe` with `scr_server` driver module)
- **Client**: Python script (`torcs_jm_par.py`)
- **Port**: 3001 (default, UDP)
- **Tick rate**: ~50 Hz (20ms per simulation step)
- **Timeout**: if client doesn't respond within ~10ms, TORCS repeats the previous command

## torcs_jm_par.py Structure

The starter script has this core loop:

```
initialize UDP connection
while race_not_finished:
    sensor_data = receive_from_torcs()    # UDP packet with all sensors
    action = compute_driving_action(sensor_data)  # YOUR LOGIC HERE
    send_to_torcs(action)                 # steering, accel, brake, gear
```

### Two Built-in Driving Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `drive_example()` | Basic driving logic, minimal | Getting started, understanding the flow |
| `drive_modular()` | Advanced with configurable parameters | Main development target |

## Sensor Data (Input from TORCS)

These are the values received every tick:

### Position & Orientation
| Sensor | Type | Range | Description |
|--------|------|-------|-------------|
| `angle` | float | [-π, π] rad | Angle between car direction and track axis. 0 = aligned, positive = pointing right |
| `trackPos` | float | [-1, 1] | Lateral position on track. 0 = center, -1 = right edge, +1 = left edge |
| `distFromStart` | float | meters | Distance from start line along track centerline |
| `distRaced` | float | meters | Total distance raced |
| `curLapTime` | float | seconds | Current lap time |
| `lastLapTime` | float | seconds | Previous lap time |
| `racePos` | int | 1-N | Current race position |

### Speed
| Sensor | Type | Unit | Description |
|--------|------|------|-------------|
| `speedX` | float | km/h | Longitudinal speed (along car axis) |
| `speedY` | float | km/h | Lateral speed |
| `speedZ` | float | km/h | Vertical speed |

### Track Sensors (CRITICAL for driving logic)
| Sensor | Type | Description |
|--------|------|-------------|
| `track` | float[19] | Array of 19 distance sensors measuring distance to track edge. Sensors span from -90° to +90° in 10° increments. Each value = distance in meters to the track edge at that angle. **This is your primary input for corner detection and steering.** |

Sensor layout (bird's eye, car pointing up):
```
              [9] = straight ahead (0°)
         [10]              [8]
      [11]                    [7]
    [12]                        [6]
  [13]                            [5]
 [14]                              [4]
[15]                                [3]
[16]                                [2]
[17]                                [1]
[18] = -90° (hard right)    [0] = +90° (hard left)

Index:  Angle from car forward:
  0  →  -90° (far left)
  1  →  -80°
  2  →  -70°
  ...
  9  →    0° (straight ahead)
  ...
 17  →  +80°
 18  →  +90° (far right)
```

**Key insight**: When `track[9]` (straight ahead) is small, a corner is approaching. Comparing left-side sensors (`track[0..8]`) vs right-side sensors (`track[10..18]`) tells you which direction the corner goes.

### Wheel Data
| Sensor | Type | Description |
|--------|------|-------------|
| `wheelSpinVel` | float[4] | Angular spin velocity of each wheel (rad/s). Order: [front-left, front-right, rear-left, rear-right] |
| `rpm` | float | Engine RPM |
| `gear` | int | Current gear (-1 = reverse, 0 = neutral, 1-6 = forward) |

### Damage & Fuel
| Sensor | Type | Description |
|--------|------|-------------|
| `damage` | float | Accumulated damage points |
| `fuel` | float | Remaining fuel level |

### Opponents
| Sensor | Type | Description |
|--------|------|-------------|
| `opponents` | float[36] | Distance to nearest opponent in 36 angular sectors (10° each, 360° coverage) |

## Actuator Commands (Output to TORCS)

| Command | Type | Range | Description |
|---------|------|-------|-------------|
| `steer` | float | [-1, +1] | Steering. -1 = full right, +1 = full left |
| `accel` | float | [0, 1] | Throttle. 0 = none, 1 = full |
| `brake` | float | [0, 1] | Brake. 0 = none, 1 = full |
| `gear` | int | -1 to 6 | Gear selection |
| `clutch` | float | [0, 1] | Clutch (usually 0) |
| `focus` | float | [-90, 90] | Focus sensor direction (rarely used) |

## User-Configurable Parameters in torcs_jm_par.py

These are the main knobs to tune in the starter code:

| Parameter | Description | Tuning Direction |
|-----------|-------------|-----------------|
| `TARGET_SPEED` | Speed the car accelerates toward | ↑ = faster but riskier corners |
| `STEER_GAIN` | How sharply car turns into corners | ↑ = sharper turns, ↓ = smoother |
| `CENTERING_GAIN` | How aggressively car returns to center | Adjust for track positioning |
| `BRAKE_THRESHOLD` | When to start braking before corners | ↓ = earlier braking, ↑ = later (braver) |
| `GEAR_SPEEDS` | Speed thresholds for gear shifts | Match to F1 car power band |
| `ENABLE_TRACTION_CONTROL` | Prevents wheel spin on acceleration | Keep enabled for stability |

## F1 Car Physics Notes

The F1 car has very different physics from the standard TORCS cars:
- **Much higher downforce** — can corner faster but more sensitive to speed
- **More powerful engine** — higher top speed, gear ratios matter more
- **Different braking behavior** — braking too hard causes understeer (not lockup like road cars)
- **Understeer tendency** — especially when braking + turning simultaneously
- **Key lesson from other teams**: reduce acceleration to 0 when approaching corners, brake gradually (not all at once)

## Corkscrew Track Notes

The Corkscrew is a technical track with:
- Significant elevation changes (the famous corkscrew section is a steep downhill with blind corners)
- Tight hairpin-style corners
- A mix of fast straights and technical sections
- Very unforgiving on corner exits — going off track is easy

**Critical corners to optimize**: the corkscrew section itself and the final corner before the start/finish straight.
