from __future__ import annotations

import argparse
import json
import random
import socket
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from research_driver import (
    BASELINE_CONFIG_PATH,
    BEST_CONFIG_PATH,
    DriverConfig,
    DriverSetup,
    INT_FIELDS,
    TunableDriver,
    clamp,
    load_driver_setup,
    save_driver_setup,
)
from track_sectors import (
    DOCUMENTED_CORKSCREW_LAYOUT_NAME,
    get_sector_layout,
    normalize_dist_from_start,
)
from torcs_jm_par import Client, data_size


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "autoresearch"
BEST_RESULT_PATH = BEST_CONFIG_PATH.with_name("autoresearch_best_result.json")

SEARCH_SPACE: dict[str, tuple[float, float, float]] = {
    "target_speed_straight": (170.0, 290.0, 12.0),
    "target_speed_corner": (35.0, 110.0, 6.0),
    "steer_gain": (8.0, 24.0, 1.0),
    "centering_gain": (0.10, 1.20, 0.08),
    "brake_distance_threshold": (100.0, 260.0, 12.0),
    "brake_intensity_max": (0.20, 1.00, 0.06),
    "corner_detect_threshold": (90.0, 220.0, 10.0),
    "gear_up_rpm": (6500.0, 10500.0, 250.0),
    "gear_down_rpm": (2500.0, 6500.0, 200.0),
    "tc_slip_threshold": (1.0, 10.0, 0.6),
    "tc_throttle_cut": (0.15, 0.95, 0.05),
    "stuck_speed_threshold": (2.0, 15.0, 0.8),
    "stuck_ticks": (20.0, 200.0, 12.0),
    "low_speed_boost_accel": (0.20, 1.00, 0.05),
    "low_speed_boost_threshold": (4.0, 30.0, 1.5),
    "turn_angle_threshold": (0.03, 0.35, 0.03),
    "in_turn_brake_cap": (0.00, 0.40, 0.03),
    "accel_divisor": (20.0, 120.0, 6.0),
    "accel_bias": (0.05, 0.80, 0.05),
    "brake_divisor": (40.0, 180.0, 8.0),
    "far_brake_factor": (0.05, 0.80, 0.05),
    "corner_throttle_cap": (0.10, 0.80, 0.05),
}

STRATEGY_GLOBAL = "global"
STRATEGY_DOCUMENTED_TURNS = "documented-turns"
GLOBAL_SEARCH_FIELDS = tuple(SEARCH_SPACE.keys())
SECTOR_SEARCH_FIELDS = (
    "target_speed_straight",
    "target_speed_corner",
    "brake_distance_threshold",
    "brake_intensity_max",
    "corner_detect_threshold",
    "in_turn_brake_cap",
    "accel_bias",
    "far_brake_factor",
    "corner_throttle_cap",
)


@dataclass
class TrialResult:
    trial_index: int
    config_path: str
    completed_lap: bool
    lap_time: float | None
    lap_source: str | None
    dist_raced: float
    max_dist_from_start: float
    current_lap_time: float
    max_speed: float
    mean_speed: float
    damage: float
    max_abs_track_pos: float
    offtrack_ticks: int
    backwards_ticks: int
    slow_ticks: int
    steps: int
    termination_reason: str
    exploration: float
    sector_layout: str | None
    sector_splits: dict[str, float]
    sector_times: dict[str, float]
    last_sector: str | None
    config: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["timestamp"] = datetime.now().isoformat(timespec="seconds")
        return record


TRIAL_RESULT_FIELD_NAMES = {field.name for field in fields(TrialResult)}


def trial_result_from_record(record: Mapping[str, Any]) -> TrialResult | None:
    filtered = {
        field_name: record[field_name]
        for field_name in TRIAL_RESULT_FIELD_NAMES
        if field_name in record
    }
    filtered.setdefault(
        "sector_times",
        compute_sector_times(
            filtered.get("sector_layout"),
            filtered.get("sector_splits", {}),
        ),
    )
    try:
        return TrialResult(**filtered)
    except TypeError:
        return None


class SafeTorcsClient(Client):
    """Original TORCS client protocol with safer Windows-friendly connection retries."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 3001,
        sid: str = "SCR",
        steps: int = 8000,
        connect_attempts: int = 20,
        connect_delay: float = 0.5,
    ) -> None:
        self._connect_attempts = max(1, connect_attempts)
        self._connect_delay = max(0.1, connect_delay)
        super().__init__(H=host, p=port, i=sid)
        self.maxSteps = steps

    def parse_the_command_line(self) -> None:
        return

    def setup_connection(self) -> None:
        try:
            self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except OSError as exc:
            raise RuntimeError("Could not create UDP socket for TORCS.") from exc

        self.so.settimeout(1.0)
        init_angles = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
        initmsg = f"{self.sid}(init {init_angles})"

        attempts_left = self._connect_attempts
        while attempts_left > 0:
            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
                sockdata, _addr = self.so.recvfrom(data_size)
                decoded = sockdata.decode("utf-8")
            except OSError:
                attempts_left -= 1
                print(
                    f"Waiting for TORCS on UDP {self.port}... attempts left: {attempts_left}"
                )
                time.sleep(self._connect_delay)
                continue

            if "identified" in decoded:
                print(f"Client connected on {self.port}.")
                return

        self.so.close()
        self.so = None
        raise RuntimeError(
            "Could not connect to TORCS. Start `wtorcs.exe` first, then launch this script "
            "from the second terminal in `torcs/gym_torcs`."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Karpathy-style autoresearch for TORCS: evaluate the current best setup, "
            "then keep trying small mutations and save the best lap."
        )
    )
    parser.add_argument("--trials", type=int, default=12, help="Number of candidate runs to evaluate.")
    parser.add_argument("--steps", type=int, default=8000, help="Maximum simulation steps per run.")
    parser.add_argument("--port", type=int, default=3001, help="SCR UDP port used by TORCS.")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=(STRATEGY_GLOBAL, STRATEGY_DOCUMENTED_TURNS),
        default=STRATEGY_GLOBAL,
        help=(
            "Search strategy. `global` tunes one config for the whole lap, "
            "`documented-turns` keeps a shared base config and adds local overrides "
            "for the 12 named Corkscrew sectors."
        ),
    )
    parser.add_argument(
        "--exploration",
        type=float,
        default=1.0,
        help="Initial mutation strength multiplier. Lower = safer, higher = more aggressive.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible parameter search.",
    )
    parser.add_argument(
        "--connect-attempts",
        type=int,
        default=20,
        help="How many times to retry connecting before failing.",
    )
    parser.add_argument(
        "--restart-pause",
        type=float,
        default=1.5,
        help="Seconds to wait after requesting a race restart.",
    )
    parser.add_argument(
        "--reset-best",
        action="store_true",
        help="Ignore the current best file and restart the search from baseline.",
    )
    return parser.parse_args()


def mutate_parameter_mapping(
    params: dict[str, Any],
    *,
    field_names: list[str],
    rng: random.Random,
    exploration: float,
    min_changes: int,
    max_changes: int,
    jump_probability: float,
) -> dict[str, Any]:
    if not field_names:
        return params

    mutation_count = rng.randint(min_changes, min(max_changes, len(field_names)))

    for field_name in rng.sample(field_names, mutation_count):
        lower, upper, step = SEARCH_SPACE[field_name]
        direction = -1.0 if rng.random() < 0.5 else 1.0
        scale = 0.75 + rng.random()
        candidate_value = params[field_name] + direction * step * exploration * scale

        if rng.random() < jump_probability:
            candidate_value = rng.uniform(lower, upper)

        if field_name in INT_FIELDS:
            params[field_name] = int(round(clamp(candidate_value, lower, upper)))
        else:
            params[field_name] = round(clamp(candidate_value, lower, upper), 6)
    return params


def mutate_config(
    base: DriverConfig,
    rng: random.Random,
    exploration: float,
    *,
    field_names: list[str] | None = None,
    min_changes: int = 3,
    max_changes: int = 7,
    jump_probability: float = 0.12,
) -> DriverConfig:
    params = base.to_dict()
    mutate_parameter_mapping(
        params,
        field_names=list(field_names or GLOBAL_SEARCH_FIELDS),
        rng=rng,
        exploration=exploration,
        min_changes=min_changes,
        max_changes=max_changes,
        jump_probability=jump_probability,
    )
    return DriverConfig.from_mapping(params)


def clone_sector_overrides(setup: DriverSetup) -> dict[str, dict[str, Any]]:
    return {
        sector_name: dict(values)
        for sector_name, values in setup.sector_overrides.items()
    }


def values_match(field_name: str, left: Any, right: Any) -> bool:
    if field_name in INT_FIELDS:
        return int(left) == int(right)
    return abs(float(left) - float(right)) < 1e-6


def extract_sector_override(
    base_config: DriverConfig,
    effective_values: dict[str, Any],
) -> dict[str, Any]:
    base_values = base_config.to_dict()
    override: dict[str, Any] = {}
    for field_name in SECTOR_SEARCH_FIELDS:
        candidate_value = effective_values[field_name]
        base_value = base_values[field_name]
        if values_match(field_name, candidate_value, base_value):
            continue
        override[field_name] = candidate_value
    return override


def compute_sector_times(
    sector_layout: str | None,
    sector_splits: dict[str, float],
) -> dict[str, float]:
    sector_specs = get_sector_layout(sector_layout)
    if not sector_specs or not sector_splits:
        return {}

    sector_times: dict[str, float] = {}
    previous_split = 0.0
    for sector in sector_specs:
        cumulative_split = sector_splits.get(sector.name)
        if cumulative_split is None:
            break
        cumulative_split = max(previous_split, float(cumulative_split))
        sector_times[sector.name] = round(cumulative_split - previous_split, 6)
        previous_split = cumulative_split
    return sector_times


def get_result_sector_times(result: TrialResult | None) -> dict[str, float]:
    if result is None:
        return {}
    if result.sector_times:
        return dict(result.sector_times)
    return compute_sector_times(result.sector_layout, result.sector_splits)


def get_next_sector_name(layout_name: str | None, current_sector_name: str | None) -> str | None:
    if current_sector_name is None:
        return None

    sector_specs = get_sector_layout(layout_name)
    for index, sector in enumerate(sector_specs[:-1]):
        if sector.name == current_sector_name:
            return sector_specs[index + 1].name
    return None


def build_sector_focus_weights(
    layout_name: str | None,
    reference_result: TrialResult | None,
) -> dict[str, float]:
    sector_specs = get_sector_layout(layout_name)
    if not sector_specs:
        return {}

    weights = {sector.name: 1.0 for sector in sector_specs}
    if reference_result is None:
        return weights

    sector_times = get_result_sector_times(reference_result)
    completed_sectors = [
        sector
        for sector in sector_specs
        if sector_times.get(sector.name, 0.0) > 0.0
    ]
    if completed_sectors:
        total_time = sum(sector_times[sector.name] for sector in completed_sectors)
        total_length = sum(max(1.0, sector.end - sector.start) for sector in completed_sectors)
        average_time_per_meter = total_time / max(total_length, 1.0)

        # Keep some exploration everywhere, but bias mutations toward sectors that
        # currently consume disproportionate time or where the run got stuck.
        for sector in completed_sectors:
            sector_time = sector_times[sector.name]
            sector_length = max(1.0, sector.end - sector.start)
            time_share = sector_time / max(total_time, 1e-6)
            density_ratio = (sector_time / sector_length) / max(average_time_per_meter, 1e-6)
            weights[sector.name] += time_share * 8.0
            weights[sector.name] += max(0.0, density_ratio - 1.0) * 2.5

    if not reference_result.completed_lap and reference_result.last_sector:
        weights[reference_result.last_sector] = weights.get(reference_result.last_sector, 1.0) + 2.5
        next_sector_name = get_next_sector_name(layout_name, reference_result.last_sector)
        if next_sector_name is not None:
            weights[next_sector_name] = weights.get(next_sector_name, 1.0) + 1.0

    return weights


def pick_weighted_sector_names(
    sector_names: list[str],
    focus_weights: dict[str, float],
    count: int,
    rng: random.Random,
) -> list[str]:
    remaining = list(sector_names)
    selected: list[str] = []

    for _ in range(min(count, len(remaining))):
        total_weight = sum(max(0.05, focus_weights.get(name, 1.0)) for name in remaining)
        threshold = rng.random() * total_weight
        cumulative_weight = 0.0
        chosen_index = len(remaining) - 1

        for index, sector_name in enumerate(remaining):
            cumulative_weight += max(0.05, focus_weights.get(sector_name, 1.0))
            if cumulative_weight >= threshold:
                chosen_index = index
                break

        selected.append(remaining.pop(chosen_index))

    return selected


def describe_sector_focus(
    layout_name: str | None,
    reference_result: TrialResult | None,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    if reference_result is None:
        return []

    focus_weights = build_sector_focus_weights(layout_name, reference_result)
    sector_times = get_result_sector_times(reference_result)
    ranked_sectors = sorted(
        focus_weights.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]

    description: list[dict[str, Any]] = []
    for sector_name, weight in ranked_sectors:
        entry: dict[str, Any] = {
            "sector": sector_name,
            "weight": round(weight, 4),
        }
        sector_time = sector_times.get(sector_name)
        if sector_time is not None:
            entry["sector_time"] = round(sector_time, 6)
        description.append(entry)
    return description


def mutate_sector_setup(
    base_setup: DriverSetup,
    rng: random.Random,
    exploration: float,
    *,
    focus_result: TrialResult | None = None,
) -> DriverSetup:
    next_base = base_setup.base_config
    if rng.random() < 0.30:
        next_base = mutate_config(
            next_base,
            rng,
            exploration * 0.65,
            field_names=[
                "steer_gain",
                "centering_gain",
                "gear_up_rpm",
                "gear_down_rpm",
                "tc_slip_threshold",
                "tc_throttle_cut",
                "low_speed_boost_accel",
                "low_speed_boost_threshold",
                "turn_angle_threshold",
                "accel_divisor",
                "accel_bias",
                "brake_divisor",
            ],
            min_changes=1,
            max_changes=3,
            jump_probability=0.05,
        )

    sector_specs = get_sector_layout(DOCUMENTED_CORKSCREW_LAYOUT_NAME)
    sector_names = [sector.name for sector in sector_specs]
    focus_weights = build_sector_focus_weights(
        DOCUMENTED_CORKSCREW_LAYOUT_NAME,
        focus_result,
    )
    sector_overrides = clone_sector_overrides(base_setup)
    sector_mutation_count = rng.randint(1, min(3, len(sector_names)))

    for sector_name in pick_weighted_sector_names(
        sector_names,
        focus_weights,
        sector_mutation_count,
        rng,
    ):
        effective_config = base_setup.resolve_config_for_sector(sector_name)
        effective_values = effective_config.to_dict()
        mutate_parameter_mapping(
            effective_values,
            field_names=list(SECTOR_SEARCH_FIELDS),
            rng=rng,
            exploration=exploration,
            min_changes=2,
            max_changes=4,
            jump_probability=0.08,
        )

        if rng.random() < 0.18:
            revert_field = rng.choice(SECTOR_SEARCH_FIELDS)
            effective_values[revert_field] = next_base.to_dict()[revert_field]

        override = extract_sector_override(next_base, effective_values)
        if override:
            sector_overrides[sector_name] = override
        else:
            sector_overrides.pop(sector_name, None)

    if sector_overrides and rng.random() < 0.10:
        removable_names = list(sector_overrides.keys())
        removal_weights = {
            sector_name: 1.0 / max(0.25, focus_weights.get(sector_name, 1.0))
            for sector_name in removable_names
        }
        sector_to_remove = pick_weighted_sector_names(
            removable_names,
            removal_weights,
            1,
            rng,
        )[0]
        sector_overrides.pop(sector_to_remove, None)

    candidate_setup = DriverSetup(
        base_config=next_base,
        sector_layout=DOCUMENTED_CORKSCREW_LAYOUT_NAME,
        sector_overrides=sector_overrides,
    )
    if candidate_setup.to_dict() != base_setup.to_dict():
        return candidate_setup

    forced_sector_name = pick_weighted_sector_names(
        sector_names,
        focus_weights,
        1,
        rng,
    )[0]
    forced_values = base_setup.resolve_config_for_sector(forced_sector_name).to_dict()
    mutate_parameter_mapping(
        forced_values,
        field_names=list(SECTOR_SEARCH_FIELDS),
        rng=rng,
        exploration=max(0.5, exploration),
        min_changes=2,
        max_changes=3,
        jump_probability=0.0,
    )
    sector_overrides[forced_sector_name] = extract_sector_override(next_base, forced_values)
    if not sector_overrides[forced_sector_name]:
        sector_overrides[forced_sector_name] = {
            "target_speed_corner": round(
                clamp(
                    next_base.target_speed_corner + SEARCH_SPACE["target_speed_corner"][2],
                    SEARCH_SPACE["target_speed_corner"][0],
                    SEARCH_SPACE["target_speed_corner"][1],
                ),
                6,
            )
        }
    return DriverSetup(
        base_config=next_base,
        sector_layout=DOCUMENTED_CORKSCREW_LAYOUT_NAME,
        sector_overrides=sector_overrides,
    )


def mutate_setup(
    base_setup: DriverSetup,
    rng: random.Random,
    exploration: float,
    *,
    strategy: str,
    focus_result: TrialResult | None = None,
) -> DriverSetup:
    if strategy == STRATEGY_DOCUMENTED_TURNS:
        return mutate_sector_setup(
            base_setup,
            rng,
            exploration,
            focus_result=focus_result,
        )
    return DriverSetup(
        base_config=mutate_config(base_setup.base_config, rng, exploration),
    )


def sector_balance_rank(result: TrialResult) -> tuple[float, float]:
    sector_times = sorted(
        (value for value in get_result_sector_times(result).values() if value > 0.0),
        reverse=True,
    )
    if not sector_times:
        return (float("inf"), float("inf"))
    return (
        sector_times[0],
        sum(sector_times[:2]),
    )


def result_rank(result: TrialResult) -> tuple[float, float, float, float, float, float, float]:
    sector_balance = sector_balance_rank(result)
    completed_sector_count = float(len(get_result_sector_times(result)))

    if result.completed_lap and result.lap_time is not None:
        return (
            0.0,
            result.lap_time,
            sector_balance[0],
            sector_balance[1],
            result.damage,
            float(result.offtrack_ticks),
            -result.dist_raced,
        )
    return (
        1.0,
        -result.dist_raced,
        -completed_sector_count,
        result.current_lap_time if completed_sector_count > 0 else float("inf"),
        result.damage,
        float(result.offtrack_ticks),
        result.max_abs_track_pos,
    )


def is_better(candidate: TrialResult, incumbent: TrialResult | None) -> bool:
    if incumbent is None:
        return True
    return result_rank(candidate) < result_rank(incumbent)


def request_restart(client: SafeTorcsClient, pause_seconds: float) -> None:
    if client.so is None:
        return
    client.R.d["meta"] = 1
    try:
        client.respond_to_server()
    except SystemExit:
        pass
    except OSError:
        pass
    time.sleep(max(0.0, pause_seconds))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def ensure_run_dir() -> Path:
    run_dir = RESULTS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_configs_dir(run_dir: Path) -> Path:
    configs_dir = run_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    return configs_dir


def normalize_setup_for_strategy(setup: DriverSetup, strategy: str) -> DriverSetup:
    if strategy == STRATEGY_DOCUMENTED_TURNS:
        existing_overrides = (
            clone_sector_overrides(setup)
            if setup.sector_layout == DOCUMENTED_CORKSCREW_LAYOUT_NAME
            else {}
        )
        return DriverSetup(
            base_config=setup.base_config,
            sector_layout=DOCUMENTED_CORKSCREW_LAYOUT_NAME,
            sector_overrides=existing_overrides,
        )

    return DriverSetup(base_config=setup.base_config)


def result_matches_strategy(result: TrialResult, strategy: str) -> bool:
    if strategy == STRATEGY_DOCUMENTED_TURNS:
        return result.sector_layout == DOCUMENTED_CORKSCREW_LAYOUT_NAME
    return result.sector_layout is None


def load_trial_result(path: Path) -> TrialResult | None:
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, Mapping):
        return None
    return trial_result_from_record(data)


def find_saved_global_best_result(strategy: str) -> TrialResult | None:
    best_result = load_trial_result(BEST_RESULT_PATH)
    if best_result is not None and not result_matches_strategy(best_result, strategy):
        best_result = None

    for result_path in RESULTS_ROOT.glob("*/best_result.json"):
        result = load_trial_result(result_path)
        if result is None or not result_matches_strategy(result, strategy):
            continue
        if is_better(result, best_result):
            best_result = result

    return best_result


def save_global_best_artifacts(best_setup: DriverSetup, best_result: TrialResult) -> None:
    save_driver_setup(best_setup, BEST_CONFIG_PATH)
    BEST_RESULT_PATH.write_text(
        json.dumps(best_result.to_record(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_starting_setup(reset_best: bool, strategy: str) -> DriverSetup:
    if reset_best:
        setup = load_driver_setup(BASELINE_CONFIG_PATH)
    elif BEST_CONFIG_PATH.exists():
        setup = load_driver_setup(BEST_CONFIG_PATH)
    else:
        setup = load_driver_setup(BASELINE_CONFIG_PATH)

    return normalize_setup_for_strategy(setup, strategy)


def load_global_best_state(reset_best: bool, strategy: str) -> tuple[DriverSetup, TrialResult | None]:
    if reset_best:
        return load_starting_setup(reset_best=True, strategy=strategy), None

    best_result = find_saved_global_best_result(strategy)
    if best_result is not None:
        best_setup = normalize_setup_for_strategy(
            DriverSetup.from_mapping(best_result.config),
            strategy,
        )
        return best_setup, best_result

    return load_starting_setup(reset_best=False, strategy=strategy), None


def evaluate_candidate(
    setup: DriverSetup,
    *,
    trial_index: int,
    config_path: Path,
    steps: int,
    port: int,
    connect_attempts: int,
    restart_pause: float,
    exploration: float,
) -> tuple[TrialResult, list[dict[str, Any]]]:
    client = SafeTorcsClient(
        port=port,
        steps=steps,
        connect_attempts=connect_attempts,
    )
    driver = TunableDriver(setup)
    sector_specs = get_sector_layout(setup.sector_layout)

    dist_raced = 0.0
    max_dist_from_start = 0.0
    current_lap_time = 0.0
    max_speed = 0.0
    speed_sum = 0.0
    damage = 0.0
    max_abs_track_pos = 0.0
    offtrack_ticks = 0
    backwards_ticks = 0
    slow_ticks = 0
    lap_time: float | None = None
    lap_source: str | None = None
    termination_reason = "step_limit"

    previous_dist_from_start: float | None = None
    previous_cur_lap_time = 0.0
    last_seen_last_lap_time = 0.0
    completed_steps = 0
    sector_splits: dict[str, float] = {}
    next_sector_split_index = 0
    last_sector: str | None = None
    telemetry_records: list[dict[str, Any]] = []

    try:
        for step_index in range(1, steps + 1):
            client.get_servers_input()
            if client.so is None:
                termination_reason = "server_closed"
                break

            sensors = dict(client.S.d)
            speed = float(sensors.get("speedX", 0.0))
            track_pos = float(sensors.get("trackPos", 0.0))
            angle = float(sensors.get("angle", 0.0))
            damage = float(sensors.get("damage", 0.0))
            dist_raced = float(sensors.get("distRaced", 0.0))
            dist_from_start = float(sensors.get("distFromStart", 0.0))
            normalized_dist_from_start = normalize_dist_from_start(
                setup.sector_layout,
                dist_from_start,
            )
            current_lap_time = float(sensors.get("curLapTime", 0.0))
            last_lap_time = float(sensors.get("lastLapTime", 0.0))
            track = sensors.get("track", [])

            completed_steps = step_index
            max_speed = max(max_speed, speed)
            speed_sum += speed
            max_dist_from_start = max(max_dist_from_start, dist_from_start)
            max_abs_track_pos = max(max_abs_track_pos, abs(track_pos))
            last_sector = setup.active_sector_name(dist_from_start)

            while (
                next_sector_split_index < len(sector_specs) - 1
                and normalized_dist_from_start >= sector_specs[next_sector_split_index].end
            ):
                sector_name = sector_specs[next_sector_split_index].name
                sector_splits.setdefault(sector_name, round(current_lap_time, 6))
                next_sector_split_index += 1

            if abs(track_pos) > 1.0 or (track and min(track) < 0):
                offtrack_ticks += 1
            if speed < 5.0 and step_index > 250:
                slow_ticks += 1
            else:
                slow_ticks = max(0, slow_ticks - 1)
            if angle != 0.0 and speed > 20.0 and abs(angle) > 1.57:
                backwards_ticks += 1

            should_stop = False
            if last_lap_time > 0.0 and last_lap_time != last_seen_last_lap_time:
                lap_time = last_lap_time
                lap_source = "lastLapTime"
                termination_reason = "lap_completed"
                should_stop = True
            elif (
                previous_dist_from_start is not None
                and previous_cur_lap_time > 5.0
                and dist_from_start + 50.0 < previous_dist_from_start
                and dist_raced > previous_dist_from_start + 200.0
            ):
                lap_time = previous_cur_lap_time
                lap_source = "distFromStart_wrap"
                termination_reason = "lap_completed"
                should_stop = True
            elif offtrack_ticks > 150:
                termination_reason = "too_many_offtrack_ticks"
                should_stop = True
            elif slow_ticks > 250:
                termination_reason = "stalled"
                should_stop = True
            elif backwards_ticks > 80:
                termination_reason = "driving_backwards"
                should_stop = True

            action: dict[str, Any] | None = None
            if not should_stop:
                driver.drive_client(client)
                action = dict(client.R.d)

            telemetry_records.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "step": step_index,
                    "trial_index": trial_index,
                    "sector_name": last_sector,
                    "sensors": sensors,
                    "action": action,
                    "lap_completed": lap_time is not None,
                    "lap_time": lap_time,
                    "lap_source": lap_source,
                    "termination_reason": termination_reason if should_stop else None,
                }
            )

            if should_stop:
                break

            client.respond_to_server()

            previous_dist_from_start = dist_from_start
            previous_cur_lap_time = current_lap_time
            last_seen_last_lap_time = last_lap_time
    finally:
        request_restart(client, restart_pause)
        client.shutdown()

    if lap_time is not None and sector_specs:
        sector_splits[sector_specs[-1].name] = round(lap_time, 6)

    sector_times = compute_sector_times(setup.sector_layout, sector_splits)
    mean_speed = speed_sum / completed_steps if completed_steps else 0.0
    return TrialResult(
        trial_index=trial_index,
        config_path=str(config_path),
        completed_lap=lap_time is not None,
        lap_time=lap_time,
        lap_source=lap_source,
        dist_raced=dist_raced,
        max_dist_from_start=max_dist_from_start,
        current_lap_time=current_lap_time,
        max_speed=max_speed,
        mean_speed=mean_speed,
        damage=damage,
        max_abs_track_pos=max_abs_track_pos,
        offtrack_ticks=offtrack_ticks,
        backwards_ticks=backwards_ticks,
        slow_ticks=slow_ticks,
        steps=completed_steps,
        termination_reason=termination_reason,
        exploration=exploration,
        sector_layout=setup.sector_layout,
        sector_splits=sector_splits,
        sector_times=sector_times,
        last_sector=last_sector,
        config=setup.to_dict(),
    ), telemetry_records


def persist_trial(history_path: Path, result: TrialResult) -> None:
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.to_record(), sort_keys=True) + "\n")


def persist_records(history_path: Path, records: list[dict[str, Any]]) -> None:
    with history_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(json_safe(record), sort_keys=True) + "\n")


def write_summary(
    summary_path: Path,
    *,
    best_result: TrialResult | None,
    best_setup: DriverSetup,
    best_config_path: Path,
    best_telemetry_path: Path | None,
    best_telemetry_trial_index: int | None,
    global_best_result: TrialResult | None,
    args: argparse.Namespace,
    run_dir: Path,
) -> None:
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "best_result": None if best_result is None else best_result.to_record(),
        "best_config": best_setup.to_dict(),
        "sector_focus": describe_sector_focus(best_setup.sector_layout, best_result),
        "best_config_path": str(best_config_path),
        "best_telemetry_path": None if best_telemetry_path is None else str(best_telemetry_path),
        "best_telemetry_trial_index": best_telemetry_trial_index,
        "global_best_result": None if global_best_result is None else global_best_result.to_record(),
        "global_best_config_path": str(BEST_CONFIG_PATH) if global_best_result is not None else None,
        "replay_best_command": (
            f"python record_best_run.py --config-path \"{best_config_path}\" --steps {args.steps}"
        ),
        "args": vars(args),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_best_run_artifacts(
    run_dir: Path,
    *,
    best_setup: DriverSetup,
    best_result: TrialResult | None,
    best_telemetry_records: list[dict[str, Any]] | None,
    args: argparse.Namespace,
) -> Path:
    best_config_path = run_dir / "best_config.json"
    save_driver_setup(best_setup, best_config_path)

    best_result_path = run_dir / "best_result.json"
    payload = None if best_result is None else best_result.to_record()
    best_result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if best_telemetry_records is not None:
        best_telemetry_path = run_dir / "best_telemetry.jsonl"
        persist_records(best_telemetry_path, best_telemetry_records)

    replay_command_path = run_dir / "replay_best_command.txt"
    replay_command_path.write_text(
        (
            "cd C:\\Projekty\\IBM_RACING_LEAGUE\\torcs\\gym_torcs\n"
            f"python record_best_run.py --config-path \"{best_config_path}\" --steps {args.steps}\n"
        ),
        encoding="utf-8",
    )
    return best_config_path


def print_result(result: TrialResult, prefix: str = "") -> None:
    lap_part = (
        f"lap={result.lap_time:.3f}s via {result.lap_source}"
        if result.completed_lap and result.lap_time is not None
        else f"progress={result.dist_raced:.1f}m"
    )
    slowest_sector_part = ""
    sector_times = get_result_sector_times(result)
    if sector_times:
        slowest_sector_name, slowest_sector_time = max(
            sector_times.items(),
            key=lambda item: item[1],
        )
        slowest_sector_part = f", slowest_sector={slowest_sector_name}:{slowest_sector_time:.3f}s"
    sector_part = f", last_sector={result.last_sector}" if result.last_sector else ""
    print(
        f"{prefix}{lap_part}, damage={result.damage:.1f}, offtrack={result.offtrack_ticks}, "
        f"mean_speed={result.mean_speed:.1f} km/h, reason={result.termination_reason}"
        f"{sector_part}{slowest_sector_part}"
    )


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    run_dir = ensure_run_dir()
    configs_dir = ensure_configs_dir(run_dir)
    history_path = run_dir / "trials.jsonl"
    summary_path = run_dir / "summary.json"
    best_telemetry_path = run_dir / "best_telemetry.jsonl"

    global_best_setup, global_best_result = load_global_best_state(args.reset_best, args.strategy)
    if global_best_result is not None:
        save_global_best_artifacts(global_best_setup, global_best_result)

    run_best_setup = global_best_setup
    run_best_result: TrialResult | None = None
    best_telemetry_trial_index: int | None = None
    best_config_path = write_best_run_artifacts(
        run_dir,
        best_setup=run_best_setup,
        best_result=run_best_result,
        best_telemetry_records=None,
        args=args,
    )
    current_exploration = max(0.35, args.exploration)

    print("Autoresearch starting.")
    print(f"Run dir: {run_dir}")
    print(f"Best config file: {BEST_CONFIG_PATH}")
    print(f"Strategy: {args.strategy}")
    if global_best_setup.sector_layout:
        print(
            f"Sector layout: {global_best_setup.sector_layout} "
            f"({len(get_sector_layout(global_best_setup.sector_layout))} sectors)"
        )
        print("Sector-aware search now biases mutations toward the slowest or failing sectors.")
    if global_best_result is not None:
        print_result(global_best_result, prefix="Current global best: ")
    print(
        "Reminder: run `wtorcs.exe` first in `torcs/torcs`, then launch this script "
        "from a second terminal in `torcs/gym_torcs`."
    )

    for trial_index in range(0, args.trials + 1):
        is_baseline_trial = trial_index == 0
        exploration = current_exploration
        candidate_setup = (
            global_best_setup
            if is_baseline_trial
            else mutate_setup(
                global_best_setup,
                rng,
                exploration,
                strategy=args.strategy,
                focus_result=global_best_result,
            )
        )
        candidate_config_path = configs_dir / f"trial_{trial_index:03d}.json"
        save_driver_setup(candidate_setup, candidate_config_path)

        label = "baseline/current best" if is_baseline_trial else f"candidate {trial_index}/{args.trials}"
        print(f"\nEvaluating {label} with exploration={exploration:.3f}...")

        try:
            result, telemetry_records = evaluate_candidate(
                candidate_setup,
                trial_index=trial_index,
                config_path=candidate_config_path,
                steps=args.steps,
                port=args.port,
                connect_attempts=args.connect_attempts,
                restart_pause=args.restart_pause,
                exploration=exploration,
            )
        except RuntimeError as exc:
            print(str(exc))
            write_summary(
                summary_path,
                best_result=run_best_result,
                best_setup=run_best_setup,
                best_config_path=best_config_path,
                best_telemetry_path=best_telemetry_path if best_telemetry_path.exists() else None,
                best_telemetry_trial_index=best_telemetry_trial_index,
                global_best_result=global_best_result,
                args=args,
                run_dir=run_dir,
            )
            return 1

        persist_trial(history_path, result)
        print_result(result, prefix="  ")

        run_improved = is_better(result, run_best_result)
        global_improved = is_better(result, global_best_result)

        if run_improved:
            run_best_result = result
            run_best_setup = candidate_setup
            best_telemetry_trial_index = result.trial_index
            best_config_path = write_best_run_artifacts(
                run_dir,
                best_setup=run_best_setup,
                best_result=run_best_result,
                best_telemetry_records=telemetry_records,
                args=args,
            )

        if global_improved:
            global_best_result = result
            global_best_setup = candidate_setup
            save_global_best_artifacts(global_best_setup, global_best_result)
            print(f"  -> New global best saved. Telemetry: {best_telemetry_path}")
            if not is_baseline_trial:
                current_exploration = min(2.5, current_exploration * 1.05)
        elif run_improved:
            print("  -> New run best. Global best unchanged.")
            if not is_baseline_trial:
                current_exploration = max(0.35, current_exploration * 0.92)
        else:
            print("  -> No improvement.")
            if not is_baseline_trial:
                current_exploration = max(0.35, current_exploration * 0.92)

        write_summary(
            summary_path,
            best_result=run_best_result,
            best_setup=run_best_setup,
            best_config_path=best_config_path,
            best_telemetry_path=best_telemetry_path if best_telemetry_path.exists() else None,
            best_telemetry_trial_index=best_telemetry_trial_index,
            global_best_result=global_best_result,
            args=args,
            run_dir=run_dir,
        )

    print("\nAutoresearch finished.")
    if run_best_result is not None:
        print_result(run_best_result, prefix="Run best: ")
    if global_best_result is not None:
        print_result(global_best_result, prefix="Global best: ")
        print(f"Global best parameters saved to: {BEST_CONFIG_PATH}")
    print(f"Full history saved to: {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
