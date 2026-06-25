import json
import pickle
import numpy as np
from pathlib import Path
from snakeoil import Client
from mk_driver import TunableDriver, load_driver_setup

def get_obs(S):
    """Same format as in torcs_env._get_obs()"""
    track_sensors = S.get('track', [0]*19)
    if len(track_sensors) < 19:
        track_sensors = [0]*19

    speed_x   = np.clip(S.get('speedX', 0) / 300.0, -1, 1)
    speed_y   = np.clip(S.get('speedY', 0) / 50.0,  -1, 1)
    speed_z   = np.clip(S.get('speedZ', 0) / 50.0,  -1, 1)
    track_pos = np.clip(S.get('trackPos', 0), -1, 1)
    angle     = np.clip(S.get('angle', 0) / np.pi, -1, 1)
    track_norm = np.clip(np.array(track_sensors, dtype=np.float32) / 200.0, 0, 1)
    wheel_spin = S.get('wheelSpinVel', [0,0,0,0])
    if len(wheel_spin) < 4:
        wheel_spin = [0,0,0,0]
    wheel_norm = np.clip(np.array(wheel_spin, dtype=np.float32) / 100.0, -1, 1)
    rpm  = np.clip(S.get('rpm', 0) / 10000.0, 0, 1)
    gear = np.clip(S.get('gear', 0) / 6.0, -1, 1)

    return np.concatenate([
        [speed_x, speed_y, speed_z],
        [track_pos],
        [angle],
        track_norm,
        wheel_norm,
        [rpm],
        [gear],
    ]).astype(np.float32)

def action_to_network_space(accel, brake, steer):
    """
    Converts expert actions [0,1] to agent action_space format [-1,1]
    inverse of what _apply_action() does
    """
    net = accel - brake
    # net > 0 → action[1] = net rescaled to [-1,1]
    # net < 0 → action[1] negative
    action_accel = net * 2.0 - 1.0  # [0,1] → [-1,1]
    action_steer = np.clip(steer, -1, 1)
    return np.array([action_steer, action_accel], dtype=np.float32)

def collect(config_path='trial_078.json', laps=3, port=3001):
    setup = load_driver_setup(config_path)
    driver = TunableDriver(setup)
    client = Client(p=port)

    collected = []
    lap_count = 0
    prev_dist = 0

    print(f"Collecting expert data — {laps} laps...")

    for step in range(client.maxSteps, 0, -1):
        client.get_servers_input()
        S = client.S.d

        # Observation before action
        obs = get_obs(S)

        # Expert generates action
        driver.drive_client(client)

        # Get generated actions
        accel = client.R.d['accel']
        brake = client.R.d['brake']
        steer = client.R.d['steer']

        # Send to TORCS
        client.respond_to_server()

        # Get new state
        client.get_servers_input()
        S_next = client.S.d
        next_obs = get_obs(S_next)

        # Convert actions to agent format
        action = action_to_network_space(accel, brake, steer)

        # Simple reward
        speed     = S.get('speedX', 0)
        angle     = S.get('angle', 0)
        track_pos = S.get('trackPos', 0)
        reward    = speed * np.cos(angle) / 300.0 - 0.5 * track_pos**2

        # Detect lap
        dist = S_next.get('distFromStart', 0)
        if prev_dist > 2000 and dist < 200:
            lap_count += 1
            print(f"Lap {lap_count}/{laps} | collected {len(collected)} steps")
            if lap_count >= laps:
                break
        prev_dist = dist

        collected.append({
            'obs':      obs,
            'action':   action,
            'reward':   reward,
            'next_obs': next_obs,
            'done':     False,
        })

    client.shutdown()

    # Save data
    output_path = 'expert_data.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(collected, f)

    print(f"Saved {len(collected)} steps to {output_path}")
    return collected

if __name__ == '__main__':
    collect(config_path='autoresearch_best.json', laps=40)