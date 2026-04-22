from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from reward import RewardBreakdown, compute_reward
from rl_torcs_client import RLTorcsClient, request_race_restart


OBSERVATION_SIZE = 32


@dataclass(frozen=True)
class TorcsEnvConfig:
    host: str = "localhost"
    port: int = 3001
    connect_attempts: int = 20
    connect_delay: float = 0.5
    restart_pause: float = 1.5
    max_episode_steps: int = 9000
    offtrack_track_pos: float = 1.0
    offtrack_ticks_limit: int = 60
    damage_limit: float = 5000.0
    stuck_speed_threshold: float = 1.0
    stuck_ticks_limit: int = 250
    stuck_grace_steps: int = 400
    backwards_angle_threshold: float = 1.57
    backwards_ticks_limit: int = 80
    min_lap_time_for_wrap: float = 5.0
    lap_wrap_margin: float = 50.0
    lap_completion_bonus: float = 25.0
    lap_completion_factor: float = 200.0
    failure_penalty: float = -5.0
    launch_assist_steps: int = 250
    launch_speed_threshold: float = 15.0
    launch_accel_min: float = 0.35

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TorcsRLEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, config: TorcsEnvConfig | None = None) -> None:
        super().__init__()
        self.config = config or TorcsEnvConfig()

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self.client: RLTorcsClient | None = None
        self.previous_sensors: dict[str, Any] | None = None
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_torcs_action: dict[str, Any] = {
            "steer": 0.0,
            "accel": 0.0,
            "brake": 0.0,
            "gear": 1,
            "clutch": 0.0,
            "meta": 0,
        }
        self.last_observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        self.last_seen_last_lap_time = 0.0

        self.episode_step = 0
        self.episode_reward = 0.0
        self.max_speed = 0.0
        self.speed_sum = 0.0
        self.max_abs_track_pos = 0.0
        self.offtrack_ticks = 0
        self.stuck_ticks = 0
        self.backwards_ticks = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._restart_connection()

        assert self.client is not None
        self.client.get_servers_input()
        if self.client.so is None:
            raise RuntimeError("TORCS closed the connection during reset.")

        sensors = dict(self.client.S.d)
        self.previous_sensors = sensors
        self.last_seen_last_lap_time = float(sensors.get("lastLapTime", 0.0))
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_torcs_action = {
            "steer": 0.0,
            "accel": 0.0,
            "brake": 0.0,
            "gear": int(float(sensors.get("gear", 1))) if sensors.get("gear") is not None else 1,
            "clutch": 0.0,
            "meta": 0,
        }
        self.episode_step = 0
        self.episode_reward = 0.0
        self.max_speed = float(sensors.get("speedX", 0.0))
        self.speed_sum = float(sensors.get("speedX", 0.0))
        self.max_abs_track_pos = abs(float(sensors.get("trackPos", 0.0)))
        self.offtrack_ticks = 0
        self.stuck_ticks = 0
        self.backwards_ticks = 0
        self.last_observation = self._build_observation(sensors)
        return self.last_observation.copy(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.client is None or self.client.so is None or self.previous_sensors is None:
            raise RuntimeError("Environment is not connected. Call reset() first.")

        action_array = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        action_array = np.clip(action_array, self.action_space.low, self.action_space.high)

        self._apply_action(action_array)
        self.client.respond_to_server()
        self.client.get_servers_input()

        if self.client.so is None:
            reward = self.config.failure_penalty
            self.episode_reward += reward
            info = {
                "termination_reason": "server_closed",
                "raw_sensors": dict(self.previous_sensors or {}),
                "torcs_action": dict(self.last_torcs_action),
                "policy_action": self.last_action.tolist(),
                "episode_summary": self._build_episode_summary(
                    lap_completed=False,
                    lap_time=None,
                    lap_source=None,
                    termination_reason="server_closed",
                    reward_breakdown=RewardBreakdown(
                        total=reward,
                        progress_reward=0.0,
                        speed_reward=0.0,
                        center_penalty=0.0,
                        angle_penalty=0.0,
                        damage_penalty=0.0,
                        lateral_penalty=0.0,
                        offtrack_penalty=0.0,
                        backward_penalty=0.0,
                        lap_bonus=0.0,
                        terminal_penalty=reward,
                    ),
                ),
            }
            return self.last_observation.copy(), reward, True, False, info

        sensors = dict(self.client.S.d)
        self.episode_step += 1
        self.max_speed = max(self.max_speed, float(sensors.get("speedX", 0.0)))
        self.speed_sum += float(sensors.get("speedX", 0.0))
        self.max_abs_track_pos = max(
            self.max_abs_track_pos, abs(float(sensors.get("trackPos", 0.0)))
        )

        self._update_failure_counters(sensors)
        lap_completed, lap_time, lap_source = self._detect_lap_completion(sensors)

        terminated = False
        truncated = False
        termination_reason = "running"
        lap_bonus = 0.0
        terminal_penalty = 0.0

        if lap_completed and lap_time is not None:
            lap_bonus = self.config.lap_completion_bonus + (
                self.config.lap_completion_factor / max(lap_time, 1.0)
            )
            terminated = True
            termination_reason = "lap_completed"
        elif self.offtrack_ticks > self.config.offtrack_ticks_limit:
            terminated = True
            termination_reason = "too_many_offtrack_ticks"
            terminal_penalty = self.config.failure_penalty
        elif self.stuck_ticks > self.config.stuck_ticks_limit:
            terminated = True
            termination_reason = "stalled"
            terminal_penalty = self.config.failure_penalty
        elif self.backwards_ticks > self.config.backwards_ticks_limit:
            terminated = True
            termination_reason = "driving_backwards"
            terminal_penalty = self.config.failure_penalty
        elif float(sensors.get("damage", 0.0)) > self.config.damage_limit:
            terminated = True
            termination_reason = "damage_limit"
            terminal_penalty = self.config.failure_penalty
        elif self.episode_step >= self.config.max_episode_steps:
            truncated = True
            termination_reason = "step_limit"

        reward_breakdown = compute_reward(
            self.previous_sensors,
            sensors,
            lap_bonus=lap_bonus,
            terminal_penalty=terminal_penalty,
        )
        reward = float(reward_breakdown.total)
        self.episode_reward += reward

        self.last_action = action_array
        self.previous_sensors = sensors
        self.last_seen_last_lap_time = float(sensors.get("lastLapTime", 0.0))
        observation = self._build_observation(sensors)
        self.last_observation = observation

        info = {
            "lap_completed": lap_completed,
            "lap_time": lap_time,
            "lap_source": lap_source,
            "termination_reason": termination_reason,
            "raw_sensors": dict(sensors),
            "torcs_action": dict(self.last_torcs_action),
            "policy_action": action_array.tolist(),
            "reward_breakdown": reward_breakdown.to_dict(),
        }
        if terminated or truncated:
            info["episode_summary"] = self._build_episode_summary(
                lap_completed=lap_completed,
                lap_time=lap_time,
                lap_source=lap_source,
                termination_reason=termination_reason,
                reward_breakdown=reward_breakdown,
            )

        return observation.copy(), reward, terminated, truncated, info

    def close(self) -> None:
        if self.client is not None:
            self.client.shutdown()
            self.client = None

    def _restart_connection(self) -> None:
        if self.client is not None:
            request_race_restart(self.client, self.config.restart_pause)
            self.client.shutdown()
            self.client = None

        self.client = RLTorcsClient(
            host=self.config.host,
            port=self.config.port,
            steps=self.config.max_episode_steps,
            connect_attempts=self.config.connect_attempts,
            connect_delay=self.config.connect_delay,
        )

    def _apply_action(self, action: np.ndarray) -> None:
        assert self.client is not None
        sensors = self.previous_sensors or {}
        response = self.client.R.d

        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))
        if throttle_brake >= 0.0:
            accel = throttle_brake
            brake = 0.0
        else:
            accel = 0.0
            brake = -throttle_brake

        speed = float(sensors.get("speedX", 0.0))
        # Standing starts are especially hard for a fresh policy; a small launch assist
        # avoids wasting most early training on learning "do not hold the brakes at spawn".
        if (
            self.episode_step < self.config.launch_assist_steps
            and speed < self.config.launch_speed_threshold
        ):
            accel = max(accel, self.config.launch_accel_min)
            brake = 0.0

        response["steer"] = steer
        response["accel"] = accel
        response["brake"] = brake
        response["gear"] = self._auto_gear(sensors)
        response["clutch"] = 0.0
        response["meta"] = 0
        self.last_torcs_action = {
            "steer": response["steer"],
            "accel": response["accel"],
            "brake": response["brake"],
            "gear": response["gear"],
            "clutch": response["clutch"],
            "meta": response["meta"],
        }

    def _auto_gear(self, sensors: dict[str, Any]) -> int:
        gear = int(float(sensors.get("gear", 1)))
        rpm = float(sensors.get("rpm", 0.0))

        if gear <= 0:
            return 1
        if rpm > 8500 and gear < 6:
            return gear + 1
        if rpm < 4000 and gear > 1:
            return gear - 1
        return gear

    def _build_observation(self, sensors: dict[str, Any]) -> np.ndarray:
        track = np.clip(
            np.asarray(sensors.get("track", [0.0] * 19), dtype=np.float32) / 200.0,
            -1.0,
            1.0,
        )
        wheel_spin = np.clip(
            np.asarray(sensors.get("wheelSpinVel", [0.0] * 4), dtype=np.float32) / 200.0,
            -1.0,
            1.0,
        )
        obs = np.concatenate(
            [
                np.asarray(
                    [
                        np.clip(float(sensors.get("angle", 0.0)) / math.pi, -1.0, 1.0),
                        np.clip(float(sensors.get("trackPos", 0.0)) / 1.5, -1.0, 1.0),
                        np.clip(float(sensors.get("speedX", 0.0)) / 300.0, -1.0, 1.0),
                        np.clip(float(sensors.get("speedY", 0.0)) / 100.0, -1.0, 1.0),
                        np.clip(float(sensors.get("speedZ", 0.0)) / 100.0, -1.0, 1.0),
                        np.clip(float(sensors.get("rpm", 0.0)) / 10000.0, 0.0, 1.0),
                        np.clip(float(sensors.get("damage", 0.0)) / 5000.0, 0.0, 1.0),
                        self.last_action[0],
                        self.last_action[1],
                    ],
                    dtype=np.float32,
                ),
                track,
                wheel_spin,
            ]
        ).astype(np.float32)
        return obs

    def _update_failure_counters(self, sensors: dict[str, Any]) -> None:
        speed = float(sensors.get("speedX", 0.0))
        angle = float(sensors.get("angle", 0.0))
        track_pos = float(sensors.get("trackPos", 0.0))
        track = sensors.get("track", [])

        is_offtrack = abs(track_pos) > self.config.offtrack_track_pos or (track and min(track) < 0)
        if is_offtrack:
            self.offtrack_ticks += 1
        else:
            self.offtrack_ticks = max(0, self.offtrack_ticks - 1)

        if (
            speed < self.config.stuck_speed_threshold
            and self.episode_step > self.config.stuck_grace_steps
        ):
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = max(0, self.stuck_ticks - 1)

        if speed > 20.0 and abs(angle) > self.config.backwards_angle_threshold:
            self.backwards_ticks += 1
        else:
            self.backwards_ticks = max(0, self.backwards_ticks - 1)

    def _detect_lap_completion(self, sensors: dict[str, Any]) -> tuple[bool, float | None, str | None]:
        last_lap_time = float(sensors.get("lastLapTime", 0.0))
        if last_lap_time > 0.0 and last_lap_time != self.last_seen_last_lap_time:
            return True, last_lap_time, "lastLapTime"

        if self.previous_sensors is None:
            return False, None, None

        previous_dist = float(self.previous_sensors.get("distFromStart", 0.0))
        previous_cur_lap = float(self.previous_sensors.get("curLapTime", 0.0))
        current_dist = float(sensors.get("distFromStart", 0.0))
        current_raced = float(sensors.get("distRaced", 0.0))

        if (
            previous_cur_lap > self.config.min_lap_time_for_wrap
            and current_dist + self.config.lap_wrap_margin < previous_dist
            and current_raced > previous_dist + 200.0
        ):
            return True, previous_cur_lap, "distFromStart_wrap"

        return False, None, None

    def _build_episode_summary(
        self,
        *,
        lap_completed: bool,
        lap_time: float | None,
        lap_source: str | None,
        termination_reason: str,
        reward_breakdown: RewardBreakdown,
    ) -> dict[str, Any]:
        sensors = self.previous_sensors or {}
        mean_speed = self.speed_sum / max(1, self.episode_step + 1)
        return {
            "episode_steps": self.episode_step,
            "episode_reward": self.episode_reward,
            "lap_completed": lap_completed,
            "lap_time": lap_time,
            "lap_source": lap_source,
            "termination_reason": termination_reason,
            "dist_raced": float(sensors.get("distRaced", 0.0)),
            "current_lap_time": float(sensors.get("curLapTime", 0.0)),
            "damage": float(sensors.get("damage", 0.0)),
            "max_speed": self.max_speed,
            "mean_speed": mean_speed,
            "max_abs_track_pos": self.max_abs_track_pos,
            "offtrack_ticks": self.offtrack_ticks,
            "stuck_ticks": self.stuck_ticks,
            "backwards_ticks": self.backwards_ticks,
            "reward_breakdown": reward_breakdown.to_dict(),
        }
