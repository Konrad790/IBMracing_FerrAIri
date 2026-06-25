from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from track_sectors import get_sector_name
from snakeoil import PI


MODULE_DIR = Path(__file__).resolve().parent
BASELINE_CONFIG_PATH = MODULE_DIR / "autoresearch_baseline.json"
BEST_CONFIG_PATH = MODULE_DIR / "autoresearch_best.json"

INT_FIELDS = {"gear_up_rpm", "gear_down_rpm", "stuck_ticks"}


@dataclass(frozen=True)
class DriverConfig:
    target_speed_straight: float = 200.0
    target_speed_corner: float = 50.0
    steer_gain: float = 15.0
    centering_gain: float = 0.50
    brake_distance_threshold: float = 200.0
    brake_intensity_max: float = 0.70
    corner_detect_threshold: float = 150.0
    gear_up_rpm: int = 8000
    gear_down_rpm: int = 4000
    tc_slip_threshold: float = 5.0
    tc_throttle_cut: float = 0.50
    stuck_speed_threshold: float = 5.0
    stuck_ticks: int = 100
    low_speed_boost_accel: float = 0.50
    low_speed_boost_threshold: float = 10.0
    turn_angle_threshold: float = 0.10
    in_turn_brake_cap: float = 0.10
    accel_divisor: float = 50.0
    accel_bias: float = 0.30
    brake_divisor: float = 100.0
    far_brake_factor: float = 0.30
    corner_throttle_cap: float = 0.40

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DriverConfig":
        valid = {field.name for field in fields(cls)}
        cleaned: dict[str, Any] = {}
        for key, value in data.items():
            if key not in valid:
                continue
            cleaned[key] = normalize_config_value(key, value)
        return cls(**cleaned)


@dataclass(frozen=True)
class DriverSetup:
    base_config: DriverConfig
    sector_layout: str | None = None
    sector_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        if self.sector_layout is None and not self.sector_overrides:
            return self.base_config.to_dict()
        return {
            "schema_version": 2,
            "base_config": self.base_config.to_dict(),
            "sector_layout": self.sector_layout,
            "sector_overrides": self.sector_overrides,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DriverSetup":
        if "base_config" not in data and "sector_layout" not in data and "sector_overrides" not in data:
            return cls(base_config=DriverConfig.from_mapping(data))

        base_config = DriverConfig.from_mapping(data.get("base_config", {}))
        sector_layout_value = data.get("sector_layout")
        sector_layout = str(sector_layout_value) if sector_layout_value else None

        valid_fields = {field_info.name for field_info in fields(DriverConfig)}
        raw_sector_overrides = data.get("sector_overrides", {})
        cleaned_sector_overrides: dict[str, dict[str, Any]] = {}
        if isinstance(raw_sector_overrides, Mapping):
            for sector_name, raw_values in raw_sector_overrides.items():
                if not isinstance(raw_values, Mapping):
                    continue
                cleaned_values: dict[str, Any] = {}
                for key, value in raw_values.items():
                    if key not in valid_fields:
                        continue
                    cleaned_values[key] = normalize_config_value(key, value)
                if cleaned_values:
                    cleaned_sector_overrides[str(sector_name)] = cleaned_values

        return cls(
            base_config=base_config,
            sector_layout=sector_layout,
            sector_overrides=cleaned_sector_overrides,
        )

    def active_sector_name(self, dist_from_start: float) -> str | None:
        return get_sector_name(self.sector_layout, dist_from_start)

    def resolve_config_for_sector(self, sector_name: str | None) -> DriverConfig:
        if sector_name is None:
            return self.base_config
        overrides = self.sector_overrides.get(sector_name)
        if not overrides:
            return self.base_config

        params = self.base_config.to_dict()
        params.update(overrides)
        return DriverConfig.from_mapping(params)

    def resolve_config_for_distance(self, dist_from_start: float) -> DriverConfig:
        return self.resolve_config_for_sector(self.active_sector_name(dist_from_start))


def normalize_config_value(field_name: str, value: Any) -> Any:
    if field_name in INT_FIELDS:
        return int(round(float(value)))
    return float(value)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def load_driver_setup(path: str | Path) -> DriverSetup:
    config_path = Path(path)
    if not config_path.exists():
        return DriverSetup(base_config=DriverConfig())
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return DriverSetup.from_mapping(data)


def save_driver_setup(setup: DriverSetup, path: str | Path) -> Path:
    config_path = Path(path)
    config_path.write_text(
        json.dumps(setup.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return config_path


def load_driver_config(path: str | Path) -> DriverConfig:
    return load_driver_setup(path).base_config


def save_driver_config(config: DriverConfig, path: str | Path) -> Path:
    return save_driver_setup(DriverSetup(base_config=config), path)


class TunableDriver:
    def __init__(self, setup: DriverSetup | DriverConfig) -> None:
        if isinstance(setup, DriverConfig):
            setup = DriverSetup(base_config=setup)
        self.setup = setup
        self.stuck_counter = 0
        self.last_sector_name: str | None = None

    def compute_target_speed(self, config: DriverConfig, track_sensors: list[float]) -> float:
        distance_ahead = track_sensors[9]

        if distance_ahead > config.brake_distance_threshold:
            return config.target_speed_straight

        ratio = distance_ahead / config.brake_distance_threshold
        speed = config.target_speed_corner + ratio * (
            config.target_speed_straight - config.target_speed_corner
        )
        return max(config.target_speed_corner, speed)

    def compute_steering(self, config: DriverConfig, angle: float, track_pos: float) -> float:
        steer = angle * config.steer_gain / PI
        steer -= track_pos * config.centering_gain
        return clamp(steer, -1.0, 1.0)

    def compute_throttle_brake(
        self,
        config: DriverConfig,
        speed: float,
        target_speed: float,
        angle: float,
        track_sensors: list[float],
    ) -> tuple[float, float]:
        distance_ahead = track_sensors[9]
        speed_excess = speed - target_speed
        is_turning = abs(angle) > config.turn_angle_threshold

        accel = 0.0
        brake = 0.0

        if speed_excess > 0:
            if is_turning:
                brake = min(config.in_turn_brake_cap, speed_excess / 200.0)
            else:
                brake_strength = min(
                    config.brake_intensity_max,
                    speed_excess / config.brake_divisor,
                )
                if distance_ahead < config.brake_distance_threshold:
                    proximity = 1.0 - distance_ahead / config.brake_distance_threshold
                    brake = brake_strength * proximity
                else:
                    brake = brake_strength * config.far_brake_factor
        else:
            speed_deficit = target_speed - speed
            accel = min(1.0, speed_deficit / config.accel_divisor + config.accel_bias)

            if distance_ahead < config.corner_detect_threshold:
                accel = min(accel, config.corner_throttle_cap)

        if speed < config.low_speed_boost_threshold:
            accel = max(accel, config.low_speed_boost_accel)
            brake = 0.0

        return clamp(accel, 0.0, 1.0), clamp(brake, 0.0, 1.0)

    def compute_gear(self, config: DriverConfig, current_gear: float, rpm: float) -> int:
        gear = int(current_gear)
        if gear <= 0:
            return 1
        if rpm > config.gear_up_rpm and gear < 6:
            return gear + 1
        if rpm < config.gear_down_rpm and gear > 1:
            return gear - 1
        return gear

    def apply_traction_control(
        self,
        config: DriverConfig,
        accel: float,
        wheel_spin_vel: list[float],
    ) -> float:
        rear_spin = wheel_spin_vel[2] + wheel_spin_vel[3]
        front_spin = wheel_spin_vel[0] + wheel_spin_vel[1]
        slip = rear_spin - front_spin
        if slip > config.tc_slip_threshold:
            accel *= config.tc_throttle_cut
        return max(0.0, accel)

    def drive_client(self, client: Any) -> None:
        sensors = client.S.d
        response = client.R.d

        dist_from_start = float(sensors.get("distFromStart", 0.0))
        active_config = self.setup.resolve_config_for_distance(dist_from_start)
        self.last_sector_name = self.setup.active_sector_name(dist_from_start)

        speed = sensors["speedX"]
        angle = sensors["angle"]
        track_pos = sensors["trackPos"]
        track_sensors = sensors["track"]
        rpm = sensors["rpm"]
        gear = sensors["gear"]
        wheel_spin = sensors["wheelSpinVel"]

        if speed < active_config.stuck_speed_threshold and gear > 0:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        if self.stuck_counter > active_config.stuck_ticks:
            response["steer"] = -angle * 2
            response["accel"] = active_config.low_speed_boost_accel
            response["brake"] = 0.0
            response["gear"] = -1
            if speed < -5:
                response["gear"] = 1
                self.stuck_counter = 0
            return

        target_speed = self.compute_target_speed(active_config, track_sensors)
        response["steer"] = self.compute_steering(active_config, angle, track_pos)
        accel, brake = self.compute_throttle_brake(
            active_config,
            speed,
            target_speed,
            angle,
            track_sensors,
        )
        response["accel"] = self.apply_traction_control(active_config, accel, wheel_spin)
        response["brake"] = brake
        response["gear"] = self.compute_gear(active_config, gear, rpm)