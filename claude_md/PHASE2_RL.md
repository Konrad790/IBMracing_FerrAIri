# Phase 2: Reinforcement Learning Upgrade

## Goal

Build an RL agent that learns to drive faster than the rule-based driver by discovering optimal racing lines, braking points, and speed profiles through training.

## Prerequisites (from Phase 1)

- [ ] Rule-based driver working reliably on Corkscrew
- [ ] Good understanding of sensor space (19 track sensors, speed, angle, trackPos, etc.)
- [ ] Baseline lap time established
- [ ] TORCS + Python communication working smoothly

## Architecture Decision

### Recommended: SAC (Soft Actor-Critic)

| Algorithm | Pros | Cons | Verdict |
|-----------|------|------|---------|
| **SAC** | Stable, handles continuous actions, entropy regularization prevents collapse | Slightly more complex | **Best choice** — most stable for TORCS |
| DDPG | Simpler, well-documented for TORCS | Brittle, sensitive to hyperparams | Good backup |
| PPO | Robust, widely used | Less sample efficient for continuous control | Consider if SAC struggles |
| TD3 | Improved DDPG | Less documented for TORCS | Alternative to DDPG |

### Action Space (Continuous)

```python
action = [steering, acceleration, brake]
# steering:     [-1, +1]
# acceleration: [0, 1]  
# brake:        [0, 1]
```

Optionally simplify to 2D: `[steering, accel_brake]` where negative = brake, positive = throttle.

### Observation Space

Recommended observation vector (from sensor data):

```python
observation = [
    angle,              # 1 value: car angle relative to track
    trackPos,           # 1 value: lateral position
    speedX / 300.0,     # 1 value: normalized longitudinal speed
    speedY / 50.0,      # 1 value: normalized lateral speed
    *track_sensors,     # 19 values: distance to edges (normalize by /200)
    rpm / 10000.0,      # 1 value: normalized RPM
    *wheelSpinVel,      # 4 values: wheel spin velocities (normalize)
]
# Total: ~27 dimensions
```

Normalize everything to roughly [-1, 1] or [0, 1] range.

## Reward Shaping

This is the MOST critical part. Bad reward = agent that doesn't learn.

### Recommended Reward Function

```python
def compute_reward(sensors, prev_sensors):
    # Primary: reward forward progress (speed along track)
    speed_reward = sensors.speedX / 300.0  # normalized, ~0 to 1
    
    # Penalty: being off-center (gentle, don't force center-only driving)
    center_penalty = -0.1 * abs(sensors.trackPos)
    
    # Penalty: going off track
    if abs(sensors.trackPos) > 1.0:
        off_track_penalty = -5.0
    else:
        off_track_penalty = 0.0
    
    # Penalty: damage
    damage_penalty = -0.01 * (sensors.damage - prev_sensors.damage)
    
    # Penalty: going backward
    if sensors.speedX < 0:
        backward_penalty = -1.0
    else:
        backward_penalty = 0.0
    
    # Bonus: lap completion
    lap_bonus = 0.0
    if sensors.lastLapTime > 0 and sensors.lastLapTime != prev_sensors.lastLapTime:
        lap_bonus = 100.0 / sensors.lastLapTime  # faster lap = bigger bonus
    
    reward = speed_reward + center_penalty + off_track_penalty + damage_penalty + backward_penalty + lap_bonus
    return reward
```

### Episode Termination Conditions

```python
done = (
    abs(sensors.trackPos) > 1.5 or     # way off track
    sensors.damage > 10000 or            # too much damage
    sensors.speedX < -5 or               # going backward
    stuck_counter > 200                   # stuck (speed < 5 km/h for 200 ticks)
)
```

## Environment Wrapper

### Option A: Use IBM's gym_torcs folder directly

Wrap `torcs_jm_par.py`'s communication logic into a Gym-compatible interface:

```python
import gymnasium as gym
import numpy as np

class TorcsEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=np.array([-1, 0, 0]), 
            high=np.array([1, 1, 1]), 
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(27,), dtype=np.float32
        )
        # Initialize UDP connection from torcs_jm_par.py
        self.client = TorcsClient()  # extract from torcs_jm_par.py
    
    def reset(self, seed=None):
        # Restart race in TORCS
        self.client.restart_race()
        obs = self._get_obs()
        return obs, {}
    
    def step(self, action):
        steering, accel, brake = action
        self.client.send_command(steering, accel, brake, gear=self._auto_gear())
        sensors = self.client.receive_sensors()
        obs = self._build_obs(sensors)
        reward = self._compute_reward(sensors)
        done = self._check_done(sensors)
        return obs, reward, done, False, {}
```

### Option B: Use existing gym_torcs wrapper

```python
# If gym_torcs is importable:
from gym_torcs import TorcsEnv

env = TorcsEnv(vision=False, throttle=True)
obs = env.reset(relaunch=True)
```

## Training Setup

### SAC with Stable-Baselines3

```python
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

env = TorcsEnv()

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=100_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    learning_starts=1000,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
    tensorboard_log="./tb_logs/"
)

# Train
model.learn(
    total_timesteps=500_000,
    callback=EvalCallback(env, eval_freq=5000, n_eval_episodes=3)
)

model.save("sac_torcs_corkscrew")
```

### Training Tips

1. **Start with short episodes** — terminate early if off-track, so the agent learns basic control first
2. **Curriculum learning** — first train just to drive straight, then add corners, then optimize speed
3. **Use the rule-based driver for demonstrations** — run the Phase 1 driver, record (obs, action) pairs, use for behavioral cloning pre-training
4. **Monitor with TensorBoard** — track reward, episode length, lap times
5. **TORCS memory leak** — relaunch TORCS every ~20 episodes to prevent memory leak
6. **Training speed** — consider disabling TORCS rendering for faster training (if possible with the Windows build)

## Hybrid Approach: RL + Rule-Based Fallback

```python
def hybrid_driver(sensors, rl_model, rule_based_driver):
    """Use RL for normal driving, fall back to rules in dangerous situations."""
    
    # Safety check — if about to go off track, use rule-based
    if abs(sensors.trackPos) > 0.8:
        return rule_based_driver.compute_action(sensors)
    
    # Otherwise use RL
    obs = build_observation(sensors)
    action, _ = rl_model.predict(obs, deterministic=True)
    return action
```

## Dependencies

```
# requirements.txt
numpy
gymnasium
stable-baselines3
torch
tensorboard
pandas           # for logging
```

Install:
```bash
pip install stable-baselines3[extra] gymnasium torch tensorboard pandas
```

## File Structure (Phase 2)

```
C:\torcs\gym_torcs\
├── torcs_jm_par.py           ← original starter
├── driver.py                  ← rule-based driver (Phase 1)
├── fastest.py                 ← best rule-based params
├── torcs_env.py               ← Gym environment wrapper
├── train_sac.py               ← SAC training script
├── eval.py                    ← evaluation / inference script
├── hybrid_driver.py           ← combined RL + rule-based
├── reward.py                  ← reward function (separate for easy tuning)
├── models/
│   └── sac_torcs_corkscrew/   ← saved model checkpoints
├── tb_logs/                   ← TensorBoard logs
├── results/
│   └── lap_times.csv
└── requirements.txt
```

## Success Criteria

| Metric | Rule-Based Baseline | RL Target |
|--------|-------------------|-----------|
| Lap completion | 100% | 100% |
| Fastest lap | ~2:00 | < 1:45 |
| Consistency | ~5s variance | < 3s variance |
| Corner speed | Conservative | Learned optimal |

## References

- SCR Competition Manual: https://arxiv.org/abs/1304.1672
- SAC paper: https://arxiv.org/abs/1801.01290
- Stable-Baselines3 docs: https://stable-baselines3.readthedocs.io/
- DDPG for TORCS blog: https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html
- gym_torcs: https://github.com/ugo-nama-kun/gym_torcs
- GymTorcs (Dossa fork): https://github.com/dosssman/GymTorcs
