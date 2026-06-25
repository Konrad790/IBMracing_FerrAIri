import numpy as np
import gymnasium as gym
from gymnasium import spaces
from snakeoil import Client

class TorcsEnv(gym.Env):
    """
    Gym wrapper for TORCS based on snakeoil client.
    
    Observation space: 29 sensors (speeds, position, angle, track edges)
    Action space:      3 continuous values (throttle, brake, steer)
    """

    # Number of distance sensors from track edges
    TRACK_SENSORS = 19

    def __init__(self, port=3001, vision=False):
        super().__init__()
        self.port = port
        self.vision = vision
        self.client = None        # Connection to TORCS — created in reset()
        self.terminal_judge_start = 100   # From which step we check if car is stuck
        self.time_step = 0

        # === ACTION SPACE ===
        # Agent controls three values, each in range [-1, 1]
        # Index 0: steer  — steering   (-1 = full left, +1 = full right)
        # Index 1: accel  — throttle   (-1 = none, +1 = full throttle)*
        # Index 2: brake  — brake      (-1 = none, +1 = full brake)*
        # (* we rescale to [0,1] when sending to TORCS)
        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )

        # === OBSERVATION SPACE ===
        # All values normalized to [-1, 1] or [0, 1]
        # Details in _get_obs() method
        obs_dim = 30  # explained in detail below
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Starts a new episode.
        Creates (or restarts) connection to TORCS and returns first state.
        """
        super().reset(seed=seed)

        if self.client is None or self.client.so is None:
            # First run — create client (it will launch TORCS itself)
            self.client = Client(p=self.port, vision=self.vision)
        else:
            # Next episode — send meta=1 so TORCS resets the race
            self.client.R.d['meta'] = 1
            self.client.respond_to_server()
            self.client.R.d['meta'] = 0

        self.time_step = 0
        self._stuck_count = 0
        self._prev_damage = 0
        self._prev_steer = 0.0
        self._prev_track_pos = 0.0
        self._prev_accel = 0.0
        self._prev_dist = None

        # Get first state from server
        self.client.get_servers_input()
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.time_step += 1

        # 1. Send action to TORCS
        self._apply_action(action)
        self.client.respond_to_server()

        # 2. Get new state
        self.client.get_servers_input()

        # Check if client disconnected (***restart*** or ***shutdown***)
        if self.client.so is None:
            # TORCS disconnected — end episode
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, -1.0, True, False, {}

        S = self.client.S.d


        # 3. Calculate reward
        reward = self._compute_reward(S, action)

        # 4. Check if episode ended
        terminated = self._is_terminal(S)
        truncated = False


        obs = self._get_obs()
        info = {
            'speed': S.get('speedX', 0),
            'trackPos': S.get('trackPos', 0),
            'angle': S.get('angle', 0),
            'distFromStart': S.get('distFromStart', 0),
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.client:
            self.client.shutdown()
            self.client = None





    # =========================================================
    # Helper methods — we'll fill them in next steps
    # =========================================================





    #==========================================================
    # GET OBSERVATION
    #==========================================================

    def _get_obs(self):
        """
        Converts raw TORCS data to normalized observation vector.
        
        Each value is rescaled to [-1, 1] or [0, 1] so that
        the neural network receives values on a similar scale.
        """
        S = self.client.S.d

        # Speeds — normalize by maximum sensible values
        # speedX: forward, max ~300 km/h in TORCS
        # speedY: lateral, max ~50 km/h (slip)
        # speedZ: vertical, max ~50 km/h (bumps)
        speed_x = np.clip(S.get('speedX', 0) / 300.0, -1, 1)
        speed_y = np.clip(S.get('speedY', 0) / 50.0,  -1, 1)
        speed_z = np.clip(S.get('speedZ', 0) / 50.0,  -1, 1)

        # Position on track — already in [-1, 1], but clip in case
        # of values off track (agent flies off — can be >1 or <-1)
        track_pos = np.clip(S.get('trackPos', 0), -1, 1)

        # Angle between car and track axis — in radians, range [-π, π]
        # Divide by π to get [-1, 1]
        angle = np.clip(S.get('angle', 0) / np.pi, -1, 1)

        # 19 distance sensors from edges — values in meters (0 to ~200m)
        # Normalize by 200, clip to [0, 1]
        track_sensors = S.get('track', [0] * 19)
        if len(track_sensors) < 19:
            track_sensors = [0] * 19   # safeguard in case of error
        track_norm = np.clip(
            np.array(track_sensors, dtype=np.float32) / 200.0,
            0, 1
        )

        # Wheel spin — normalize by 100 rad/s (max at full throttle)
        # Differences between wheels tell agent about slip
        wheel_spin = S.get('wheelSpinVel', [0, 0, 0, 0])
        if len(wheel_spin) < 4:
            wheel_spin = [0, 0, 0, 0]
        wheel_norm = np.clip(
            np.array(wheel_spin, dtype=np.float32) / 100.0,
            -1, 1
        )

        # Engine RPM — max ~10000 rpm in TORCS
        rpm = np.clip(S.get('rpm', 0) / 10000.0, 0, 1)

        # Gear — from -1 (reverse) to 6, normalize to [-1, 1]
        gear = np.clip(S.get('gear', 0) / 6.0, -1, 1)

        # Assemble everything into one vector
        obs = np.concatenate([
            [speed_x, speed_y, speed_z],   # 3
            [track_pos],                    # 1
            [angle],                        # 1
            track_norm,                     # 19
            wheel_norm,                     # 4
            [rpm],                          # 1
            [gear],                         # 1
        ]).astype(np.float32)               # total: 30

        return obs

    
    #==========================================================
    # APPLY ACTION
    #==========================================================


    def _apply_action(self, action):
        S = self.client.S.d
        R = self.client.R.d

        angle = S.get('angle', 0)
        track_pos = S.get('trackPos', 0)
        
        pace = np.clip(action[1], -1, 1)
        if pace >= 0:
            # Throttle floor only at start — doesn't block TC on straight
            # if S.get('speedX', 0) < self._LAUNCH_SPEED_THRESHOLD:
            #     pace = max(pace, self._LAUNCH_MIN_ACCEL)
            # pace = self.apply_traction_control(S, pace)
            R['accel'] = pace
            R['brake'] = 0
        else:
            R['accel'] = 0
            R['brake'] = -pace

        
        R['steer'] = np.clip(action[0], -1, 1)
        R['gear'] = self._auto_gear(S)
        R['clutch'] = 0
        R['meta'] = 0
    
    def apply_traction_control(self, S, accel):
        """
        TC for rear-wheel drive: rear − front (like snakeoil / mk_driver).
        Smooth throttle damping instead of ×0.5 jump; at crawl speed we ignore sensor noise.
        """
        speed = S.get('speedX', 0)
        if speed < self._TC_SKIP_BELOW_SPEED:
            return max(0.0, accel)

        wheel_spin = S.get('wheelSpinVel', [0, 0, 0, 0])
        if len(wheel_spin) < 4:
            wheel_spin = [0, 0, 0, 0]

        rear_spin = (wheel_spin[2] + wheel_spin[3]) / 2.0
        front_spin = (wheel_spin[0] + wheel_spin[1]) / 2.0
        slip = rear_spin - front_spin

        if slip > self._TC_SLIP_THRESHOLD:
            excess = min(
                1.0,
                (slip - self._TC_SLIP_THRESHOLD) / self._TC_SLIP_RANGE,
            )
            factor = max(self._TC_MIN_FACTOR, 1.0 - 0.5 * excess)
            accel *= factor

        return max(0.0, accel)


    def _auto_gear(self, S):
        """
        Automatic gearbox based on speed.
        Simple heuristic — agent doesn't waste resources learning gears.
        """
        speed = S.get('speedX', 0)
        gear = int(S.get('gear', 1))

        # Upshift thresholds
        up_thresholds   = [60, 100, 140, 180, 220]
        # Downshift thresholds (slightly lower to avoid oscillation)
        down_thresholds = [40,  80, 120, 160, 200]

        if gear < 6 and speed > up_thresholds[gear - 1]:
            return gear + 1
        elif gear > 1 and speed < down_thresholds[gear - 2]:
            return gear - 1

        return max(1, gear)  # never return to 0 (neutral)



    #==========================================================
    # COMPUTE REWARD
    #==========================================================

    # Threshold |angle| (rad): above we treat section as turn — lower penalty for steering
    _ANGLE_TURN_RAD = 0.35

    # Traction control (like mk_driver, with smoothing)
    _TC_SLIP_THRESHOLD = 5.0
    _TC_SLIP_RANGE = 20.0
    _TC_MIN_FACTOR = 0.2
    _TC_SKIP_BELOW_SPEED = 15.0
    _LAUNCH_SPEED_THRESHOLD = 30.0
    _LAUNCH_MIN_ACCEL = 0.2

    def _straight_steer_weight(self, S):
        """
        Weight of penalty for jerking steering on straight sections.

        We don't use only track[9]: during slalom the car is diagonal,
        so the sensor in the body axis sees closer to the edge, even though the track ahead
        is wide. Hence max from fan 7–11, |angle| and speedY.
        """
        track = S.get('track', [0] * 19)
        if len(track) < 19:
            track = [0] * 19

        forward_m = max(track[7], track[8], track[9], track[10], track[11])
        forward_clear = min(1.0, forward_m / 200.0)

        angle = S.get('angle', 0)
        alignment = max(0.0, 1.0 - abs(angle) / self._ANGLE_TURN_RAD)

        speed_y = abs(S.get('speedY', 0))
        lateral = min(1.0, speed_y / 15.0)

        base = forward_clear * alignment
        return base * (1.0 + 1.5 * lateral)

    def _compute_reward(self, S, action):
        """
        Reward adapted to BC / expert, with anti-slalom.

        Components:
            + speed along track axis (speed * cos(angle))
            + progress (distFromStart)
            - position on track, angle, lateral speed
            - trackPos change (slalom)
            - abrupt steering (weight from _straight_steer_weight, not just track[9])
            - |steer * throttle| on straights (slalom at full throttle)
            - collision
        """
        speed = S.get('speedX', 0)
        speed_y = S.get('speedY', 0)
        angle = S.get('angle', 0)
        track_pos = S.get('trackPos', 0)
        damage = S.get('damage', 0)
        steer_cmd = float(np.clip(action[0], -1.0, 1.0))
        pace = float(np.clip(action[1], -1.0, 1.0))

        reward_speed = np.cos(angle) * speed / 300.0

        penalty_angle = abs(angle / np.pi)
        penalty_pos = abs(track_pos)

        penalty_speed_y = abs(speed_y / 50.0)

        penalty_pos_delta = abs(track_pos - self._prev_track_pos) / 2.0
        self._prev_track_pos = track_pos


        prev_steer = getattr(self, '_prev_steer', 0)
        steer_delta = abs((steer_cmd - prev_steer) / 2.0)
        self._prev_steer = steer_cmd
        straight_weight = self._straight_steer_weight(S)
        penalty_nsmooth_steer = straight_weight * steer_delta

        penalty_steer_accel = abs(steer_cmd * pace)

        damage_delta = max(0.0, damage - self._prev_damage)
        self._prev_damage = damage
        penalty_damage = damage_delta * 0.1

        dist = float(S.get('distFromStart', 0))
        if self._prev_dist is None:
            progress = 0.0
        else:
            progress = max(0.0, dist - self._prev_dist) / 2.0
            if progress > 50.0:
                progress = 0.0
        self._prev_dist = dist

        reward = (
            0.8 * reward_speed
            + 0.4 * progress
            - 0.3 * penalty_pos
            - 0.1 * penalty_angle #0.4
            - 0.8 * penalty_pos_delta
            - 0.4 * penalty_speed_y
            - 0.2 * penalty_steer_accel #0.4
            - 0.2 * penalty_nsmooth_steer #0.3
            - 1.0 * penalty_damage
        )  # crappy, turns poorly and slaloms

        return float(reward)


    #==========================================================
    # IS TERMINAL
    #==========================================================

    def _is_terminal(self, S):
        """
        Checks if episode should end.
        
        Three conditions:
            1. Off track          (trackPos outside [-1, 1])
            2. Car stuck          (low speed for many steps)
            3. Serious collision  (damage exceeded threshold)
        """
        speed     = S.get('speedX', 0)
        track_pos = S.get('trackPos', 0)
        damage    = S.get('damage', 0)


        if abs(track_pos) > 1.3:
            return True 

        # --- Condition 2: car stuck ---
        # Check only after terminal_judge_start steps
        # to give agent time to accelerate at start
        if self.time_step > self.terminal_judge_start:
            if speed < 1.0:   # below 5 km/h = practically stopped
                self._stuck_count = getattr(self, '_stuck_count', 0) + 1
            else:
                self._stuck_count = 0

            # End only when stopped for 30 consecutive steps
            # (~0.6 seconds) — one bad step is not a problem yet
            if self._stuck_count > 30:
                #print(f"[TERMINAL] Car stuck: speed={speed:.1f} for {self._stuck_count} steps")
                self._stuck_count = 0
                return True

        # --- Condition 3: serious collision ---
        if damage > 5000:
            #print(f"[TERMINAL] Collision: damage={damage:.0f}")
            return True

        return False
