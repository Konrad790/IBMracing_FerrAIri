# Phase 1: Rule-Based AI Driver

## Goal

Build a reliable, fast rule-based AI driver for the Corkscrew track with the F1 car. This is the foundation — get consistent lap completion first, then optimize for speed.

## Development Methodology

Follow an **implement → test → refine** cycle:
1. Implement ONE rule/change at a time
2. Test by running 3–5 laps on Corkscrew
3. Record lap times and notes
4. Keep changes that improve times, revert those that don't
5. Commit working versions to git

## Step-by-Step Development Plan

### Step 1: Baseline Run (Day 1)

Run `torcs_jm_par.py` with default parameters on Corkscrew. Record:
- Does the car complete a lap? 
- What is the lap time?
- Where does the car struggle? (specific corners, straights, etc.)
- Does it go off-track? Where?

Save this as your **baseline**. Everything is compared against this.

### Step 2: Understand the Starter Code (Day 1-2)

Read `torcs_jm_par.py` thoroughly. Map out:
- Which sensors are used in `drive_example()` and `drive_modular()`
- How steering is computed (usually `angle` + `trackPos` based)
- How throttle/brake decisions are made
- How gears are shifted
- What the configurable parameters do

Create a commented version of the key functions.

### Step 3: Basic Corner Detection (Day 2-3)

Implement a function to detect approaching corners using `track` sensors:

```python
def is_corner_approaching(track_sensors, threshold=100):
    """
    Check if a corner is ahead by looking at the straight-ahead sensor.
    track_sensors[9] = distance to edge straight ahead.
    If this distance is small, we're approaching a corner.
    """
    distance_ahead = track_sensors[9]
    return distance_ahead < threshold

def get_corner_direction(track_sensors):
    """
    Determine if corner goes left or right.
    Compare sum of left sensors vs right sensors.
    """
    left_sum = sum(track_sensors[10:19])   # sensors pointing left
    right_sum = sum(track_sensors[0:9])    # sensors pointing right
    if left_sum > right_sum:
        return "left"   # more space on left = corner goes left
    else:
        return "right"
```

### Step 4: Speed Management (Day 3-4)

The most impactful optimization. Core rules:

```
IF straight detected (track[9] > threshold):
    → accelerate toward high TARGET_SPEED
IF corner approaching (track[9] < threshold):
    → reduce acceleration (possibly to 0)
    → apply graduated braking (NOT full brake)
IF in corner (trackPos drifting, angle large):
    → maintain moderate speed
    → focus on steering accuracy
```

**Critical for F1 car**: Do NOT brake hard while turning. The F1 car understeers badly with combined braking + steering. Brake BEFORE the corner, then coast/light throttle through it.

### Step 5: Steering Logic (Day 4-5)

Basic steering formula:
```python
steer = (angle * STEER_GAIN) + (trackPos * CENTERING_GAIN)
steer = max(-1, min(1, steer))  # clamp to [-1, 1]
```

Where:
- `angle` component: points the car along the track direction
- `trackPos` component: pushes the car back toward center

Tuning tips:
- If car oscillates (weaves): reduce STEER_GAIN
- If car drifts to edges: increase CENTERING_GAIN
- If car cuts corners too tight: adjust CENTERING_GAIN bias

### Step 6: Graduated Braking (Day 5-6)

Instead of binary brake on/off, implement proportional braking:

```python
def compute_brake(speed, distance_ahead, target_speed_for_corner):
    if distance_ahead > 150:
        return 0.0  # no braking on straight
    
    # How much do we need to slow down?
    speed_excess = speed - target_speed_for_corner
    
    if speed_excess <= 0:
        return 0.0  # already slow enough
    
    # Proportional braking — harder when speed excess is larger
    brake = min(1.0, speed_excess / 100.0)
    
    # Scale by distance — brake harder when corner is closer
    distance_factor = max(0.0, 1.0 - (distance_ahead / 150.0))
    
    return brake * distance_factor
```

### Step 7: Gear Management (Day 6)

Basic automatic gear shifting:

```python
GEAR_UP_SPEEDS = [0, 80, 120, 160, 200, 240, 280]  # tune for F1 car
GEAR_DOWN_SPEEDS = [0, 60, 90, 130, 170, 210, 250]

def compute_gear(speed, current_gear, rpm):
    if current_gear < 6 and speed > GEAR_UP_SPEEDS[current_gear]:
        return current_gear + 1
    elif current_gear > 1 and speed < GEAR_DOWN_SPEEDS[current_gear]:
        return current_gear - 1
    return current_gear
```

Alternatively, use RPM-based shifting (shift up at ~8000 RPM, down at ~4000 RPM — tune for the specific F1 car engine).

### Step 8: Straight Detection & Full Speed (Day 7)

Detect straights and push maximum speed:

```python
def is_straight(track_sensors, min_distance=150):
    """A straight = long distance ahead AND roughly equal distances on sides."""
    ahead = track_sensors[9]
    return ahead > min_distance

def compute_target_speed(track_sensors, base_speed, straight_speed):
    if is_straight(track_sensors):
        return straight_speed  # e.g., 300 km/h
    else:
        # Scale speed based on how tight the corner looks
        ahead = track_sensors[9]
        corner_speed = base_speed + (ahead / 200.0) * (straight_speed - base_speed)
        return max(base_speed, min(straight_speed, corner_speed))
```

### Step 9: Traction Control (Day 7-8)

Prevent wheel spin on the F1 car:

```python
def apply_traction_control(accel, wheel_spin_vel, speed):
    """Reduce throttle if rear wheels are spinning faster than car speed."""
    if speed > 10:
        # Compare wheel speed to actual car speed
        # wheel_spin_vel is in rad/s, need to convert
        rear_avg = (wheel_spin_vel[2] + wheel_spin_vel[3]) / 2
        expected_spin = speed / 3.6 / 0.33  # rough wheel radius
        slip_ratio = (rear_avg - expected_spin) / max(expected_spin, 1)
        if slip_ratio > 0.1:  # 10% slip
            accel *= 0.5  # cut throttle
    return accel
```

### Step 10: Parameter Sweep & Automation (Day 8-10)

Build a test automation system:

```python
# test_runner.py
import subprocess
import csv
import time

params_to_test = [
    {"TARGET_SPEED": 200, "STEER_GAIN": 1.0, "BRAKE_THRESHOLD": 100},
    {"TARGET_SPEED": 220, "STEER_GAIN": 1.2, "BRAKE_THRESHOLD": 80},
    # ... more combinations
]

for params in params_to_test:
    # Write params to config
    write_params(params)
    # Launch TORCS + driver
    # Wait for lap completion
    # Read lap time from output
    # Log to CSV
    log_result(params, lap_time)
```

## Key Performance Metrics

Track these after each change:

| Metric | Target |
|--------|--------|
| Lap completion rate | 100% (must complete reliably) |
| Fastest lap time | < 2:30 (baseline), aim for < 1:50 |
| Consistency | < 5s variation between laps |
| Off-track incidents | 0 per lap |
| Understeer events | Minimize (track with damage sensor) |

## Common Pitfalls (Lessons from Other Teams)

1. **Don't brake and turn simultaneously** — F1 car understeers badly
2. **Don't chase raw speed before reliability** — a completed slow lap beats a DNF
3. **The Corkscrew section needs special handling** — steep downhill + blind corners, you may need corner-specific speed limits
4. **Small changes compound** — reducing braking by 5% + increasing corner exit speed by 10 km/h adds up
5. **Test extensively** — most time should be testing, not coding

## Code Organization

```
C:\torcs\gym_torcs\
├── torcs_jm_par.py          ← original (don't modify, keep as backup)
├── driver.py                 ← your main driver (copy of torcs_jm_par.py, modified)
├── fastest.py                ← cleaned version with best-performing params
├── utils.py                  ← helper functions (corner detection, etc.)
├── test_runner.py            ← automated testing script
├── results/
│   └── lap_times.csv         ← parameter + lap time log
└── README.md
```

## When to Move to Phase 2 (RL)

Move to Phase 2 when:
- Rule-based driver reliably completes laps (100% completion rate)
- Lap time improvements have plateaued (diminishing returns from parameter tuning)
- You have a good understanding of the sensor space and physics
- You want to explore approaches that can discover non-obvious racing lines

The rule-based driver remains as:
- Baseline for comparison
- Fallback if RL training fails
- Expert demonstration data for imitation learning
