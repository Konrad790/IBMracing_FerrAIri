from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "models" / "sac_corkscrew"
RESULTS_ROOT = REPO_ROOT / "results" / "best_run_records"
BEST_LAP_SUMMARY_PATH = RESULTS_ROOT / "best_lap_summary.json"
BEST_LAP_TELEMETRY_PATH = RESULTS_ROOT / "best_lap_telemetry.jsonl"
BEST_LAP_TIME_PATH = RESULTS_ROOT / "best_lap_time.txt"
BEST_LAP_VIDEO_PATH = RESULTS_ROOT / "best_lap_video.mp4"
RL_BEST_LAP_SUMMARY_PATH = RESULTS_ROOT / "rl_best_lap_summary.json"
RL_BEST_LAP_TELEMETRY_PATH = RESULTS_ROOT / "rl_best_lap_telemetry.jsonl"
RL_BEST_LAP_TIME_PATH = RESULTS_ROOT / "rl_best_lap_time.txt"
RL_BEST_LAP_VIDEO_PATH = RESULTS_ROOT / "rl_best_lap_video.mp4"


@dataclass
class RLRunSummary:
    started_at: str
    finished_at: str
    run_dir: str
    telemetry_path: str
    model_path: str
    policy_mode: str
    episode_index: int
    completed_lap: bool
    lap_time: float | None
    lap_source: str | None
    termination_reason: str
    steps: int
    dist_raced: float
    current_lap_time: float
    damage: float
    max_speed: float
    mean_speed: float
    max_abs_track_pos: float
    total_reward: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the best RL model in TORCS and record full per-step telemetry."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_ROOT / "sac_corkscrew_best.zip"),
        help="Path to a trained SAC checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record.")
    parser.add_argument("--port", type=int, default=3001, help="SCR UDP port used by TORCS.")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=9000,
        help="Maximum number of steps per episode.",
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
        help="Seconds to wait after TORCS restart requests between episodes.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device, for example `auto`, `cpu` or `cuda`.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        type=str,
        default="ffmpeg",
        help="Path to ffmpeg used for video recording.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=30,
        help="Frame rate for recorded mp4 output.",
    )
    parser.add_argument(
        "--no-record-video",
        action="store_true",
        help="Disable mp4 recording and keep only telemetry.",
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
    dirname = f"{timestamp}_rl_{suffix}" if suffix else f"{timestamp}_rl"
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


def persist_record(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")


def load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summary_rank(summary: dict[str, Any]) -> tuple[float, float, float, float, float]:
    if summary.get("completed_lap") and summary.get("lap_time") is not None:
        return (
            0.0,
            float(summary["lap_time"]),
            float(summary.get("damage", 0.0)),
            float(summary.get("max_abs_track_pos", 0.0)),
            -float(summary.get("dist_raced", 0.0)),
        )
    return (
        1.0,
        -float(summary.get("dist_raced", 0.0)),
        float(summary.get("damage", 0.0)),
        float(summary.get("max_abs_track_pos", 0.0)),
        -float(summary.get("total_reward", 0.0)),
    )


def update_best_artifacts(summary: RLRunSummary, telemetry_path: Path, video_path: Path | None) -> bool:
    if not summary.completed_lap or summary.lap_time is None:
        return False

    summary_dict = summary.to_dict()
    existing_best = load_summary(RL_BEST_LAP_SUMMARY_PATH)
    if existing_best is not None and summary_rank(summary_dict) >= summary_rank(existing_best):
        return False

    RL_BEST_LAP_SUMMARY_PATH.write_text(
        json.dumps(summary_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    shutil.copyfile(telemetry_path, RL_BEST_LAP_TELEMETRY_PATH)
    if video_path is not None and video_path.exists():
        shutil.copyfile(video_path, RL_BEST_LAP_VIDEO_PATH)
    RL_BEST_LAP_TIME_PATH.write_text(
        "\n".join(
            [
                f"lap_time_seconds={summary.lap_time:.6f}" if summary.lap_time is not None else "lap_time_seconds=",
                f"finished_at={summary.finished_at}",
                f"run_dir={summary.run_dir}",
                f"summary_path={Path(summary.run_dir) / 'summary_episode_best.json'}",
                f"telemetry_path={RL_BEST_LAP_TELEMETRY_PATH}",
                f"video_path={RL_BEST_LAP_VIDEO_PATH}",
                f"model_path={summary.model_path}",
                f"policy_mode={summary.policy_mode}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    BEST_LAP_SUMMARY_PATH.write_text(
        json.dumps(summary_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    shutil.copyfile(telemetry_path, BEST_LAP_TELEMETRY_PATH)
    if video_path is not None and video_path.exists():
        shutil.copyfile(video_path, BEST_LAP_VIDEO_PATH)
    BEST_LAP_TIME_PATH.write_text(
        "\n".join(
            [
                f"lap_time_seconds={summary.lap_time:.6f}" if summary.lap_time is not None else "lap_time_seconds=",
                f"finished_at={summary.finished_at}",
                f"run_dir={summary.run_dir}",
                f"summary_path={Path(summary.run_dir) / 'summary_episode_best.json'}",
                f"telemetry_path={BEST_LAP_TELEMETRY_PATH}",
                f"video_path={BEST_LAP_VIDEO_PATH}",
                f"model_path={summary.model_path}",
                f"policy_mode={summary.policy_mode}",
                "source=rl",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return True


def build_summary(
    *,
    started_at: str,
    finished_at: str,
    run_dir: Path,
    telemetry_path: Path,
    model_path: Path,
    policy_mode: str,
    episode_index: int,
    episode_summary: dict[str, Any],
    total_reward: float,
) -> RLRunSummary:
    return RLRunSummary(
        started_at=started_at,
        finished_at=finished_at,
        run_dir=str(run_dir),
        telemetry_path=str(telemetry_path),
        model_path=str(model_path),
        policy_mode=policy_mode,
        episode_index=episode_index,
        completed_lap=bool(episode_summary.get("lap_completed", False)),
        lap_time=episode_summary.get("lap_time"),
        lap_source=episode_summary.get("lap_source"),
        termination_reason=str(episode_summary.get("termination_reason", "unknown")),
        steps=int(episode_summary.get("episode_steps", 0)),
        dist_raced=float(episode_summary.get("dist_raced", 0.0)),
        current_lap_time=float(episode_summary.get("current_lap_time", 0.0)),
        damage=float(episode_summary.get("damage", 0.0)),
        max_speed=float(episode_summary.get("max_speed", 0.0)),
        mean_speed=float(episode_summary.get("mean_speed", 0.0)),
        max_abs_track_pos=float(episode_summary.get("max_abs_track_pos", 0.0)),
        total_reward=total_reward,
    )


def main() -> int:
    args = parse_args()

    try:
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise SystemExit(
            "Missing RL dependencies. Install them with:\n"
            "pip install -r C:\\Projekty\\IBM_RACING_LEAGUE\\torcs\\gym_torcs\\rl_requirements.txt"
        ) from exc

    from torcs_env import TorcsEnvConfig, TorcsRLEnv
    from screen_recorder import FfmpegScreenRecorder, ScreenRecorderConfig

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model file does not exist: {model_path}")

    record_video = not args.no_record_video
    screen_recorder = FfmpegScreenRecorder(
        ScreenRecorderConfig(
            ffmpeg_path=args.ffmpeg_path,
            fps=args.video_fps,
        )
    )
    if record_video and not screen_recorder.is_available():
        raise SystemExit(
            "Automatic lap video recording requires ffmpeg on PATH.\n"
            "Install ffmpeg or rerun with `--no-record-video`."
        )

    run_dir = ensure_run_dir(args.tag)
    run_summary_path = run_dir / "episodes.jsonl"
    best_episode_summary_path = run_dir / "summary_episode_best.json"
    best_episode_telemetry_path: Path | None = None
    best_episode_video_path = run_dir / "summary_episode_best.mp4"
    best_summary: RLRunSummary | None = None
    best_video_path: Path | None = None
    policy_mode = "stochastic" if args.stochastic else "deterministic"

    env = TorcsRLEnv(
        TorcsEnvConfig(
            port=args.port,
            max_episode_steps=args.max_episode_steps,
            connect_attempts=args.connect_attempts,
            restart_pause=args.restart_pause,
        )
    )
    model = SAC.load(str(model_path), device=args.device)

    print(f"Recording RL runs with model: {model_path}")
    print(f"Run dir: {run_dir}")
    print(
        "Reminder: start `wtorcs.exe` first in `torcs/torcs`, click New Race, "
        "then launch this script from a second terminal in `torcs/gym_torcs`."
    )

    try:
        for episode_index in range(1, args.episodes + 1):
            telemetry_path = run_dir / f"episode_{episode_index:03d}_telemetry.jsonl"
            episode_video_path = run_dir / f"episode_{episode_index:03d}.mp4"
            started_at = datetime.now().isoformat(timespec="seconds")
            observation, _ = env.reset()
            if record_video:
                screen_recorder.start(episode_video_path)
            terminated = False
            truncated = False
            total_reward = 0.0
            final_info: dict[str, Any] = {}
            step_index = 0

            while not (terminated or truncated):
                action, _ = model.predict(
                    observation,
                    deterministic=not args.stochastic,
                )
                observation, reward, terminated, truncated, info = env.step(action)
                step_index += 1
                total_reward += float(reward)
                final_info = dict(info)

                persist_record(
                    telemetry_path,
                    {
                        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                        "episode_index": episode_index,
                        "step": step_index,
                        "policy_action": info.get("policy_action"),
                        "torcs_action": info.get("torcs_action"),
                        "sensors": info.get("raw_sensors"),
                        "reward": float(reward),
                        "reward_breakdown": info.get("reward_breakdown"),
                        "lap_completed": info.get("lap_completed"),
                        "lap_time": info.get("lap_time"),
                        "lap_source": info.get("lap_source"),
                        "terminated": terminated,
                        "truncated": truncated,
                    },
                )

            recorded_video_path: Path | None = None
            if record_video:
                recorded_video_path = screen_recorder.stop(keep_file=True)

            episode_summary = dict(final_info.get("episode_summary", {}))
            episode_summary["episode_index"] = episode_index
            episode_summary["policy_mode"] = policy_mode
            episode_summary["model_path"] = str(model_path)
            episode_summary["telemetry_path"] = str(telemetry_path)
            episode_summary["timestamp"] = datetime.now().isoformat(timespec="seconds")
            episode_summary["total_reward"] = total_reward
            persist_record(run_summary_path, episode_summary)

            summary = build_summary(
                started_at=started_at,
                finished_at=datetime.now().isoformat(timespec="seconds"),
                run_dir=run_dir,
                telemetry_path=telemetry_path,
                model_path=model_path,
                policy_mode=policy_mode,
                episode_index=episode_index,
                episode_summary=episode_summary,
                total_reward=total_reward,
            )

            lap_text = (
                f"{summary.lap_time:.3f}s" if summary.lap_time is not None else "n/a"
            )
            print(
                f"Episode {episode_index}: lap_completed={summary.completed_lap}, "
                f"lap_time={lap_text}, dist_raced={summary.dist_raced:.1f}, "
                f"reason={summary.termination_reason}"
            )

            if best_summary is None or summary_rank(summary.to_dict()) < summary_rank(best_summary.to_dict()):
                best_summary = summary
                best_episode_telemetry_path = telemetry_path
                best_video_path = recorded_video_path
                if (
                    record_video
                    and recorded_video_path is not None
                    and recorded_video_path.exists()
                    and summary.completed_lap
                    and summary.lap_time is not None
                ):
                    shutil.copyfile(recorded_video_path, best_episode_video_path)
            elif (
                record_video
                and recorded_video_path is not None
                and recorded_video_path.exists()
            ):
                recorded_video_path.unlink(missing_ok=True)
    finally:
        if record_video and screen_recorder.is_recording():
            leftover = screen_recorder.stop(keep_file=False)
            if leftover is not None and leftover.exists():
                leftover.unlink(missing_ok=True)
        env.close()

    if best_summary is None or best_episode_telemetry_path is None:
        print("No RL episodes were recorded.")
        return 1

    best_episode_summary_path.write_text(
        json.dumps(best_summary.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    updated_best = update_best_artifacts(best_summary, best_episode_telemetry_path, best_video_path)

    print(f"Best episode summary saved to: {best_episode_summary_path}")
    if updated_best:
        print(f"New best RL lap saved to: {RL_BEST_LAP_SUMMARY_PATH}")
        print(f"New canonical best lap saved to: {BEST_LAP_SUMMARY_PATH}")
        print(f"Best RL telemetry copied to: {RL_BEST_LAP_TELEMETRY_PATH}")
        if best_video_path is not None:
            print(f"Best RL video copied to: {RL_BEST_LAP_VIDEO_PATH}")
        print(f"Best lap time saved to: {BEST_LAP_TIME_PATH}")
    else:
        print("No improvement over the previously saved RL best lap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
