from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from autoresearch import SafeTorcsClient, compute_sector_times, request_restart
from research_driver import BEST_CONFIG_PATH, TunableDriver, load_driver_setup
from track_sectors import get_sector_layout, normalize_dist_from_start


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "best_run_records"
RULE_BASED_BEST_LAP_SUMMARY_PATH = RESULTS_ROOT / "rule_based_best_lap_summary.json"
RULE_BASED_BEST_LAP_TELEMETRY_PATH = RESULTS_ROOT / "rule_based_best_lap_telemetry.jsonl"
RULE_BASED_BEST_LAP_TIME_PATH = RESULTS_ROOT / "rule_based_best_lap_time.txt"


@dataclass
class RunSummary:
    started_at: str
    finished_at: str
    run_dir: str
    telemetry_path: str
    config_path: str
    config: dict[str, Any]
    sector_layout: str | None
    sector_splits: dict[str, float]
    sector_times: dict[str, float]
    last_sector: str | None
    completed_lap: bool
    lap_time: float | None
    lap_source: str | None
    termination_reason: str
    steps: int
    dist_raced: float
    max_dist_from_start: float
    current_lap_time: float
    max_speed: float
    mean_speed: float
    damage: float
    max_abs_track_pos: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current best rule-based setup and record full per-step telemetry "
            "for later analysis."
        )
    )
    parser.add_argument("--steps", type=int, default=8000, help="Maximum simulation steps.")
    parser.add_argument("--port", type=int, default=3001, help="SCR UDP port used by TORCS.")
    parser.add_argument(
        "--config-path",
        type=str,
        default=str(BEST_CONFIG_PATH),
        help="Path to the rule-based config JSON to replay.",
    )
    parser.add_argument(
        "--connect-attempts",
        type=int,
        default=20,
        help="How many times to retry connecting before failing.",
    )
    parser.add_argument(
        "--restart-at-end",
        action="store_true",
        help="Send a TORCS race restart after recording finishes.",
    )
    parser.add_argument(
        "--restart-pause",
        type=float,
        default=1.5,
        help="Seconds to wait after an optional restart request.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix added to the output folder name.",
    )
    return parser.parse_args()


def sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", tag.strip())
    return cleaned.strip("._-")


def ensure_run_dir(tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = sanitize_tag(tag)
    dirname = f"{timestamp}_{suffix}" if suffix else timestamp
    run_dir = RESULTS_ROOT / dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def persist_record(telemetry_path: Path, payload: dict[str, Any]) -> None:
    with telemetry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")


def load_best_lap_summary() -> dict[str, Any] | None:
    if not RULE_BASED_BEST_LAP_SUMMARY_PATH.exists():
        return None
    return json.loads(RULE_BASED_BEST_LAP_SUMMARY_PATH.read_text(encoding="utf-8"))


def current_run_is_better(summary: RunSummary, existing_best: dict[str, Any] | None) -> bool:
    if not summary.completed_lap or summary.lap_time is None:
        return False
    if existing_best is None:
        return True

    existing_completed = bool(existing_best.get("completed_lap"))
    existing_lap_time = existing_best.get("lap_time")
    if not existing_completed or existing_lap_time is None:
        return True

    return float(summary.lap_time) < float(existing_lap_time)


def update_best_lap_artifacts(summary: RunSummary, telemetry_path: Path) -> bool:
    existing_best = load_best_lap_summary()
    if not current_run_is_better(summary, existing_best):
        return False

    RULE_BASED_BEST_LAP_SUMMARY_PATH.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    shutil.copyfile(telemetry_path, RULE_BASED_BEST_LAP_TELEMETRY_PATH)
    RULE_BASED_BEST_LAP_TIME_PATH.write_text(
        "\n".join(
            [
                f"lap_time_seconds={summary.lap_time:.6f}",
                f"finished_at={summary.finished_at}",
                f"run_dir={summary.run_dir}",
                f"summary_path={Path(summary.run_dir) / 'summary.json'}",
                f"telemetry_path={RULE_BASED_BEST_LAP_TELEMETRY_PATH}",
                f"config_path={summary.config_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return True


def detect_lap_completion(
    sensors: dict[str, Any],
    *,
    previous_dist_from_start: float | None,
    previous_cur_lap_time: float,
    last_seen_last_lap_time: float,
) -> tuple[bool, float | None, str | None]:
    last_lap_time = float(sensors.get("lastLapTime", 0.0))
    if last_lap_time > 0.0 and last_lap_time != last_seen_last_lap_time:
        return True, last_lap_time, "lastLapTime"

    if previous_dist_from_start is None:
        return False, None, None

    dist_from_start = float(sensors.get("distFromStart", 0.0))
    dist_raced = float(sensors.get("distRaced", 0.0))
    if (
        previous_cur_lap_time > 5.0
        and dist_from_start + 50.0 < previous_dist_from_start
        and dist_raced > previous_dist_from_start + 200.0
    ):
        return True, previous_cur_lap_time, "distFromStart_wrap"

    return False, None, None


def build_summary(
    *,
    started_at: str,
    run_dir: Path,
    telemetry_path: Path,
    config_path: Path,
    config: dict[str, Any],
    sector_layout: str | None,
    sector_splits: dict[str, float],
    sector_times: dict[str, float],
    last_sector: str | None,
    completed_lap: bool,
    lap_time: float | None,
    lap_source: str | None,
    termination_reason: str,
    steps: int,
    dist_raced: float,
    max_dist_from_start: float,
    current_lap_time: float,
    max_speed: float,
    speed_sum: float,
    damage: float,
    max_abs_track_pos: float,
) -> RunSummary:
    mean_speed = speed_sum / steps if steps else 0.0
    return RunSummary(
        started_at=started_at,
        finished_at=datetime.now().isoformat(timespec="seconds"),
        run_dir=str(run_dir),
        telemetry_path=str(telemetry_path),
        config_path=str(config_path),
        config=config,
        sector_layout=sector_layout,
        sector_splits=sector_splits,
        sector_times=sector_times,
        last_sector=last_sector,
        completed_lap=completed_lap,
        lap_time=lap_time,
        lap_source=lap_source,
        termination_reason=termination_reason,
        steps=steps,
        dist_raced=dist_raced,
        max_dist_from_start=max_dist_from_start,
        current_lap_time=current_lap_time,
        max_speed=max_speed,
        mean_speed=mean_speed,
        damage=damage,
        max_abs_track_pos=max_abs_track_pos,
    )


def main() -> int:
    args = parse_args()
    run_dir = ensure_run_dir(args.tag)
    telemetry_path = run_dir / "telemetry.jsonl"
    summary_path = run_dir / "summary.json"
    used_config_snapshot_path = run_dir / "used_config.json"

    config_path = Path(args.config_path)
    if not config_path.exists():
        raise SystemExit(f"Config file does not exist: {config_path}")

    setup = load_driver_setup(config_path)
    used_config_snapshot_path.write_text(
        json.dumps(setup.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    driver = TunableDriver(setup)
    started_at = datetime.now().isoformat(timespec="seconds")

    client = SafeTorcsClient(
        port=args.port,
        steps=args.steps,
        connect_attempts=args.connect_attempts,
    )

    dist_raced = 0.0
    max_dist_from_start = 0.0
    current_lap_time = 0.0
    max_speed = 0.0
    speed_sum = 0.0
    damage = 0.0
    max_abs_track_pos = 0.0
    steps_completed = 0
    lap_time: float | None = None
    lap_source: str | None = None
    termination_reason = "step_limit"
    previous_dist_from_start: float | None = None
    previous_cur_lap_time = 0.0
    last_seen_last_lap_time = 0.0
    sector_specs = get_sector_layout(setup.sector_layout)
    sector_splits: dict[str, float] = {}
    next_sector_split_index = 0
    last_sector: str | None = None

    print("Recording best run telemetry.")
    print(f"Run dir: {run_dir}")
    print(f"Config path: {config_path}")
    print(
        "Reminder: start `wtorcs.exe` first in `torcs/torcs`, then run this script "
        "from a second terminal in `torcs/gym_torcs`."
    )

    try:
        for step_index in range(1, args.steps + 1):
            client.get_servers_input()
            if client.so is None:
                termination_reason = "server_closed"
                break

            sensors = dict(client.S.d)
            steps_completed = step_index
            dist_raced = float(sensors.get("distRaced", 0.0))
            dist_from_start = float(sensors.get("distFromStart", 0.0))
            normalized_dist_from_start = normalize_dist_from_start(
                setup.sector_layout,
                dist_from_start,
            )
            current_lap_time = float(sensors.get("curLapTime", 0.0))
            speed = float(sensors.get("speedX", 0.0))
            damage = float(sensors.get("damage", 0.0))
            track_pos = float(sensors.get("trackPos", 0.0))

            max_dist_from_start = max(max_dist_from_start, dist_from_start)
            max_speed = max(max_speed, speed)
            speed_sum += speed
            max_abs_track_pos = max(max_abs_track_pos, abs(track_pos))
            last_sector = setup.active_sector_name(dist_from_start)

            while (
                next_sector_split_index < len(sector_specs) - 1
                and normalized_dist_from_start >= sector_specs[next_sector_split_index].end
            ):
                sector_name = sector_specs[next_sector_split_index].name
                sector_splits.setdefault(sector_name, round(current_lap_time, 6))
                next_sector_split_index += 1

            completed_lap, detected_lap_time, detected_lap_source = detect_lap_completion(
                sensors,
                previous_dist_from_start=previous_dist_from_start,
                previous_cur_lap_time=previous_cur_lap_time,
                last_seen_last_lap_time=last_seen_last_lap_time,
            )

            driver.drive_client(client)
            action = dict(client.R.d)

            persist_record(
                telemetry_path,
                {
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "step": step_index,
                    "sector_name": driver.last_sector_name,
                    "sensors": sensors,
                    "action": action,
                    "lap_completed": completed_lap,
                    "lap_time": detected_lap_time,
                    "lap_source": detected_lap_source,
                },
            )

            if completed_lap:
                lap_time = detected_lap_time
                lap_source = detected_lap_source
                termination_reason = "lap_completed"
                break

            client.respond_to_server()
            previous_dist_from_start = dist_from_start
            previous_cur_lap_time = current_lap_time
            last_seen_last_lap_time = float(sensors.get("lastLapTime", 0.0))
    finally:
        if args.restart_at_end:
            request_restart(client, args.restart_pause)
        client.shutdown()

    if lap_time is not None and sector_specs:
        sector_splits[sector_specs[-1].name] = round(lap_time, 6)

    sector_times = compute_sector_times(setup.sector_layout, sector_splits)
    summary = build_summary(
        started_at=started_at,
        run_dir=run_dir,
        telemetry_path=telemetry_path,
        config_path=config_path,
        config=setup.to_dict(),
        sector_layout=setup.sector_layout,
        sector_splits=sector_splits,
        sector_times=sector_times,
        last_sector=last_sector,
        completed_lap=lap_time is not None,
        lap_time=lap_time,
        lap_source=lap_source,
        termination_reason=termination_reason,
        steps=steps_completed,
        dist_raced=dist_raced,
        max_dist_from_start=max_dist_from_start,
        current_lap_time=current_lap_time,
        max_speed=max_speed,
        speed_sum=speed_sum,
        damage=damage,
        max_abs_track_pos=max_abs_track_pos,
    )
    summary_path.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    is_new_best_lap = update_best_lap_artifacts(summary, telemetry_path)

    if summary.completed_lap and summary.lap_time is not None:
        print(f"Lap recorded: {summary.lap_time:.3f}s via {summary.lap_source}.")
        if is_new_best_lap:
            print(f"New best rule-based lap saved to: {RULE_BASED_BEST_LAP_SUMMARY_PATH}")
            print(f"Rule-based best telemetry copied to: {RULE_BASED_BEST_LAP_TELEMETRY_PATH}")
            print(f"Rule-based best lap time saved to: {RULE_BASED_BEST_LAP_TIME_PATH}")
    else:
        print(f"Run finished without a full lap. Reason: {summary.termination_reason}.")
    print(f"Telemetry saved to: {telemetry_path}")
    print(f"Summary saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
