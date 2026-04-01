"""
Phase 1 Rule-Based AI Driver for TORCS — Corkscrew track, F1 car.
Based on torcs_jm_par.py starter code.
"""

import math
import sys
import os

# Add parent dir so we can import the Client from the starter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torcs_jm_par import Client, PI

# ================= USER CONFIGURABLE PARAMETERS =================
# Speed management
TARGET_SPEED_STRAIGHT = 200   # km/h on straights
TARGET_SPEED_CORNER = 50      # km/h base speed in tight corners
CORNER_SPEED_FACTOR = 0.8     # multiplier: corner_speed = base + factor * distance_ahead

# Steering
STEER_GAIN = 15.0             # how aggressively car follows track angle
CENTERING_GAIN = 0.50         # how aggressively car returns to center

# Braking
BRAKE_DISTANCE_THRESHOLD = 200  # start considering braking when track[9] < this (meters)
BRAKE_INTENSITY_MAX = 0.7       # maximum brake force (graduated, never binary 1.0)

# Corner detection
CORNER_DETECT_THRESHOLD = 150   # track[9] < this = corner approaching

# Gear shifting (RPM-based)
GEAR_UP_RPM = 8000
GEAR_DOWN_RPM = 4000

# Traction control
TC_SLIP_THRESHOLD = 5.0    # wheel spin difference threshold
TC_THROTTLE_CUT = 0.5      # multiply throttle by this when slipping

# Recovery
STUCK_SPEED_THRESHOLD = 5.0
STUCK_TICKS = 100


# ================= DRIVING FUNCTIONS =================

def detect_corner(track_sensors):
    """Returns (is_corner, direction, distance_ahead, tightness)."""
    distance_ahead = track_sensors[9]
    is_corner = distance_ahead < CORNER_DETECT_THRESHOLD

    left_sum = sum(track_sensors[10:19])
    right_sum = sum(track_sensors[0:9])

    if left_sum > right_sum:
        direction = "left"
    else:
        direction = "right"

    # tightness: 0 = wide open, 1 = very tight
    tightness = max(0.0, 1.0 - distance_ahead / CORNER_DETECT_THRESHOLD)

    return is_corner, direction, distance_ahead, tightness


def compute_target_speed(track_sensors):
    """Adaptive target speed based on what's ahead."""
    distance_ahead = track_sensors[9]

    if distance_ahead > BRAKE_DISTANCE_THRESHOLD:
        # Straight — full speed
        return TARGET_SPEED_STRAIGHT

    # Scale speed based on how far ahead the edge is
    ratio = distance_ahead / BRAKE_DISTANCE_THRESHOLD
    speed = TARGET_SPEED_CORNER + ratio * (TARGET_SPEED_STRAIGHT - TARGET_SPEED_CORNER)
    return max(TARGET_SPEED_CORNER, speed)


def compute_steering(angle, track_pos, track_sensors):
    """Steering based on angle + centering + lookahead."""
    # Base: follow track direction
    steer = angle * STEER_GAIN / PI

    # Centering correction
    steer -= track_pos * CENTERING_GAIN

    return max(-1.0, min(1.0, steer))


def compute_throttle_brake(speed, target_speed, angle, track_sensors):
    """
    Returns (accel, brake).
    CRITICAL: never brake and steer hard at the same time (F1 understeer).
    """
    distance_ahead = track_sensors[9]
    speed_excess = speed - target_speed
    is_turning = abs(angle) > 0.1

    accel = 0.0
    brake = 0.0

    if speed_excess > 0:
        # Need to slow down
        if is_turning:
            # In a turn — just lift off throttle, minimal braking
            accel = 0.0
            brake = min(0.1, speed_excess / 200.0)
        else:
            # Straight — can brake harder
            brake_strength = min(BRAKE_INTENSITY_MAX, speed_excess / 100.0)
            # Scale by proximity to corner
            if distance_ahead < BRAKE_DISTANCE_THRESHOLD:
                proximity = 1.0 - distance_ahead / BRAKE_DISTANCE_THRESHOLD
                brake = brake_strength * proximity
            else:
                brake = brake_strength * 0.3
    else:
        # Need to speed up
        speed_deficit = target_speed - speed
        accel = min(1.0, speed_deficit / 50.0 + 0.3)

        # Gentle throttle in corners
        if distance_ahead < CORNER_DETECT_THRESHOLD:
            accel = min(accel, 0.4)

    # Low speed boost (avoid getting stuck from standstill)
    if speed < 10:
        accel = max(accel, 0.5)
        brake = 0.0

    return max(0.0, min(1.0, accel)), max(0.0, min(1.0, brake))


def compute_gear(speed, current_gear, rpm):
    """RPM-based gear shifting."""
    if current_gear <= 0:
        return 1

    if rpm > GEAR_UP_RPM and current_gear < 6:
        return current_gear + 1
    elif rpm < GEAR_DOWN_RPM and current_gear > 1:
        return current_gear - 1

    return current_gear


def apply_traction_control(accel, wheel_spin_vel, speed):
    """Reduce throttle if rear wheels spin faster than fronts."""
    rear_spin = wheel_spin_vel[2] + wheel_spin_vel[3]
    front_spin = wheel_spin_vel[0] + wheel_spin_vel[1]
    slip = rear_spin - front_spin

    if slip > TC_SLIP_THRESHOLD:
        accel *= TC_THROTTLE_CUT

    return max(0.0, accel)


# ================= MAIN DRIVE FUNCTION =================

stuck_counter = 0

def drive(c):
    global stuck_counter

    S = c.S.d
    R = c.R.d

    speed = S['speedX']
    angle = S['angle']
    track_pos = S['trackPos']
    track_sensors = S['track']
    rpm = S['rpm']
    gear = S['gear']
    wheel_spin = S['wheelSpinVel']

    # Stuck recovery
    if speed < STUCK_SPEED_THRESHOLD and gear > 0:
        stuck_counter += 1
    else:
        stuck_counter = 0

    if stuck_counter > STUCK_TICKS:
        R['steer'] = -angle * 2
        R['accel'] = 0.5
        R['brake'] = 0.0
        R['gear'] = -1
        if speed < -5:
            R['gear'] = 1
            stuck_counter = 0
        return

    # Normal driving
    target_speed = compute_target_speed(track_sensors)
    R['steer'] = compute_steering(angle, track_pos, track_sensors)
    accel, brake = compute_throttle_brake(speed, target_speed, angle, track_sensors)

    accel = apply_traction_control(accel, wheel_spin, speed)

    R['accel'] = accel
    R['brake'] = brake
    R['gear'] = compute_gear(speed, gear, rpm)


# ================= MAIN LOOP =================

if __name__ == "__main__":
    C = Client(p=3001)
    for step in range(C.maxSteps, 0, -1):
        C.get_servers_input()
        drive(C)
        C.respond_to_server()
    C.shutdown()
