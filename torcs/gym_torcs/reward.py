from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    progress_reward: float
    speed_reward: float
    center_penalty: float
    angle_penalty: float
    damage_penalty: float
    lateral_penalty: float
    offtrack_penalty: float
    backward_penalty: float
    lap_bonus: float
    terminal_penalty: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "progress_reward": self.progress_reward,
            "speed_reward": self.speed_reward,
            "center_penalty": self.center_penalty,
            "angle_penalty": self.angle_penalty,
            "damage_penalty": self.damage_penalty,
            "lateral_penalty": self.lateral_penalty,
            "offtrack_penalty": self.offtrack_penalty,
            "backward_penalty": self.backward_penalty,
            "lap_bonus": self.lap_bonus,
            "terminal_penalty": self.terminal_penalty,
        }


def compute_reward(
    previous_sensors: Mapping[str, Any] | None,
    sensors: Mapping[str, Any],
    *,
    lap_bonus: float = 0.0,
    terminal_penalty: float = 0.0,
) -> RewardBreakdown:
    if previous_sensors is None:
        return RewardBreakdown(
            total=0.0,
            progress_reward=0.0,
            speed_reward=0.0,
            center_penalty=0.0,
            angle_penalty=0.0,
            damage_penalty=0.0,
            lateral_penalty=0.0,
            offtrack_penalty=0.0,
            backward_penalty=0.0,
            lap_bonus=lap_bonus,
            terminal_penalty=terminal_penalty,
        )

    dist_raced = float(sensors.get("distRaced", 0.0))
    prev_dist_raced = float(previous_sensors.get("distRaced", 0.0))
    progress_delta = max(-2.0, min(2.5, dist_raced - prev_dist_raced))

    angle = float(sensors.get("angle", 0.0))
    speed_x = float(sensors.get("speedX", 0.0))
    speed_y = float(sensors.get("speedY", 0.0))
    track_pos = float(sensors.get("trackPos", 0.0))
    forward_alignment = speed_x * math.cos(angle)
    forward_speed = max(0.0, forward_alignment) / 300.0

    damage = float(sensors.get("damage", 0.0))
    prev_damage = float(previous_sensors.get("damage", 0.0))
    damage_delta = max(0.0, damage - prev_damage)

    track = sensors.get("track", [])
    is_offtrack = abs(track_pos) > 1.0 or (track and min(track) < 0)

    progress_reward = progress_delta
    speed_reward = 0.25 * forward_speed
    center_penalty = 0.12 * abs(track_pos)
    angle_penalty = 0.05 * abs(angle)
    damage_penalty = 0.003 * damage_delta
    lateral_penalty = 0.02 * min(abs(speed_y) / 50.0, 1.0)
    offtrack_penalty = 1.5 if is_offtrack else 0.0
    backward_penalty = 1.0 if forward_alignment < 0.0 else 0.0

    total = (
        progress_reward
        + speed_reward
        - center_penalty
        - angle_penalty
        - damage_penalty
        - lateral_penalty
        - offtrack_penalty
        - backward_penalty
        + lap_bonus
        + terminal_penalty
    )

    return RewardBreakdown(
        total=total,
        progress_reward=progress_reward,
        speed_reward=speed_reward,
        center_penalty=center_penalty,
        angle_penalty=angle_penalty,
        damage_penalty=damage_penalty,
        lateral_penalty=lateral_penalty,
        offtrack_penalty=offtrack_penalty,
        backward_penalty=backward_penalty,
        lap_bonus=lap_bonus,
        terminal_penalty=terminal_penalty,
    )
