from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "models" / "sac_corkscrew"
TB_ROOT = REPO_ROOT / "tb_logs" / "sac_corkscrew"
RESULTS_ROOT = REPO_ROOT / "results" / "rl_training"
BEST_RUN_RECORDS_ROOT = REPO_ROOT / "results" / "best_run_records"
RL_TRAINING_BEST_SUMMARY_PATH = BEST_RUN_RECORDS_ROOT / "rl_training_best_lap_summary.json"
RL_TRAINING_BEST_TELEMETRY_PATH = BEST_RUN_RECORDS_ROOT / "rl_training_best_lap_telemetry.jsonl"
RL_TRAINING_BEST_TIME_PATH = BEST_RUN_RECORDS_ROOT / "rl_training_best_lap_time.txt"
RL_TRAINING_BEST_VIDEO_PATH = BEST_RUN_RECORDS_ROOT / "rl_training_best_lap_video.mp4"
BEST_LAP_SUMMARY_PATH = BEST_RUN_RECORDS_ROOT / "best_lap_summary.json"
BEST_LAP_TELEMETRY_PATH = BEST_RUN_RECORDS_ROOT / "best_lap_telemetry.jsonl"
BEST_LAP_TIME_PATH = BEST_RUN_RECORDS_ROOT / "best_lap_time.txt"
BEST_LAP_VIDEO_PATH = BEST_RUN_RECORDS_ROOT / "best_lap_video.mp4"
LATEST_MODEL_PATH = MODEL_ROOT / "sac_corkscrew_latest"
BEST_MODEL_PATH = MODEL_ROOT / "sac_corkscrew_best"
LATEST_REPLAY_BUFFER_PATH = MODEL_ROOT / "sac_corkscrew_latest_replay_buffer.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SAC policy for TORCS without touching the rule-based driver."
    )
    parser.add_argument("--timesteps", type=int, default=100000, help="Total SAC training steps.")
    parser.add_argument("--port", type=int, default=3001, help="SCR UDP port used by TORCS.")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=9000,
        help="Maximum environment steps per episode before truncation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="SAC learning rate.")
    parser.add_argument(
        "--learning-rate-schedule",
        type=str,
        default="linear",
        choices=("constant", "linear", "cosine"),
        help="Learning-rate schedule used during SAC training.",
    )
    parser.add_argument(
        "--learning-rate-final",
        type=float,
        default=5e-5,
        help="Final learning rate used by non-constant schedules near the end of training.",
    )
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size.")
    parser.add_argument("--batch-size", type=int, default=256, help="SAC batch size.")
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=2000,
        help="Number of steps collected before gradient updates start.",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient.")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10000,
        help="Save a checkpoint every N environment steps.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Optional path to an existing SAC checkpoint to continue training.",
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Force a brand-new training run even if a latest checkpoint already exists.",
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
        help="Path to ffmpeg used for automatic lap video recording.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=30,
        help="Frame rate for automatic lap video recording.",
    )
    parser.add_argument(
        "--no-record-video",
        action="store_true",
        help="Disable automatic mp4 recording of training episodes.",
    )
    return parser.parse_args()


def ensure_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit(
            "Missing RL dependencies. Install them with:\n"
            "pip install -r C:\\Projekty\\IBM_RACING_LEAGUE\\torcs\\gym_torcs\\rl_requirements.txt"
        ) from exc

    return SAC, BaseCallback, CallbackList, CheckpointCallback, Monitor


def summary_rank(summary: dict[str, Any]) -> tuple[float, float, float, float, float]:
    if summary.get("lap_completed") and summary.get("lap_time") is not None:
        return (
            0.0,
            float(summary["lap_time"]),
            float(summary.get("damage", 0.0)),
            float(summary.get("offtrack_ticks", 0.0)),
            -float(summary.get("dist_raced", 0.0)),
        )

    return (
        1.0,
        -float(summary.get("dist_raced", 0.0)),
        float(summary.get("damage", 0.0)),
        float(summary.get("offtrack_ticks", 0.0)),
        float(summary.get("max_abs_track_pos", 0.0)),
    )


def infer_replay_buffer_path(model_path: Path) -> Path:
    if model_path.suffix == ".zip":
        return model_path.with_name(f"{model_path.stem}_replay_buffer.pkl")
    return model_path.with_name(f"{model_path.name}_replay_buffer.pkl")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_learning_rate_schedule(args: argparse.Namespace) -> float | Callable[[float], float]:
    initial = float(args.learning_rate)
    final = float(args.learning_rate_final)
    schedule_name = str(args.learning_rate_schedule)

    if initial <= 0.0:
        raise SystemExit("`--learning-rate` must be greater than 0.")
    if final <= 0.0:
        raise SystemExit("`--learning-rate-final` must be greater than 0.")

    if schedule_name == "constant":
        return initial

    def clamp_progress(progress_remaining: float) -> float:
        return max(0.0, min(1.0, float(progress_remaining)))

    if schedule_name == "linear":
        return lambda progress_remaining: final + (initial - final) * clamp_progress(
            progress_remaining
        )

    if schedule_name == "cosine":
        def cosine_schedule(progress_remaining: float) -> float:
            clamped_progress = clamp_progress(progress_remaining)
            completed_fraction = 1.0 - clamped_progress
            cosine_weight = 0.5 * (1.0 + math.cos(math.pi * completed_fraction))
            return final + (initial - final) * cosine_weight

        return cosine_schedule

    raise SystemExit(f"Unsupported learning-rate schedule: {schedule_name}")


def describe_learning_rate_schedule(args: argparse.Namespace) -> str:
    if args.learning_rate_schedule == "constant":
        return f"constant ({args.learning_rate:.6g})"
    return (
        f"{args.learning_rate_schedule} "
        f"({args.learning_rate:.6g} -> {args.learning_rate_final:.6g})"
    )


def resolve_learning_rate(
    learning_rate: float | Callable[[float], float], progress_remaining: float = 1.0
) -> float:
    if callable(learning_rate):
        return float(learning_rate(progress_remaining))
    return float(learning_rate)


def apply_learning_rate_to_model(
    model: Any,
    learning_rate: float | Callable[[float], float],
    *,
    progress_remaining: float = 1.0,
) -> None:
    model.learning_rate = learning_rate
    if hasattr(model, "_setup_lr_schedule"):
        model._setup_lr_schedule()

    current_learning_rate = resolve_learning_rate(learning_rate, progress_remaining)
    optimizers: list[Any] = []
    seen_optimizer_ids: set[int] = set()

    for attr_name in ("actor", "critic"):
        module = getattr(model, attr_name, None)
        optimizer = getattr(module, "optimizer", None)
        if optimizer is not None and id(optimizer) not in seen_optimizer_ids:
            optimizers.append(optimizer)
            seen_optimizer_ids.add(id(optimizer))

    ent_coef_optimizer = getattr(model, "ent_coef_optimizer", None)
    if ent_coef_optimizer is not None and id(ent_coef_optimizer) not in seen_optimizer_ids:
        optimizers.append(ent_coef_optimizer)

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_learning_rate


def main() -> int:
    args = parse_args()
    SAC, BaseCallback, CallbackList, CheckpointCallback, Monitor = ensure_dependencies()

    from torcs_env import TorcsEnvConfig, TorcsRLEnv
    from screen_recorder import FfmpegScreenRecorder, ScreenRecorderConfig

    learning_rate_schedule = build_learning_rate_schedule(args)
    run_dir = RESULTS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    TB_ROOT.mkdir(parents=True, exist_ok=True)
    BEST_RUN_RECORDS_ROOT.mkdir(parents=True, exist_ok=True)

    config = TorcsEnvConfig(port=args.port, max_episode_steps=args.max_episode_steps)
    monitor_path = run_dir / "monitor.csv"
    episodes_path = run_dir / "episodes.jsonl"
    training_summary_path = run_dir / "training_summary.json"
    best_episode_telemetry_path = run_dir / "best_episode_telemetry.jsonl"
    best_episode_video_path = run_dir / "best_episode_video.mp4"
    latest_model_path = LATEST_MODEL_PATH
    best_model_path = BEST_MODEL_PATH
    latest_model_zip_path = latest_model_path.with_suffix(".zip")
    effective_resume_path: Path | None = None
    effective_replay_buffer_path: Path | None = None
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
            "Install ffmpeg or rerun with `--no-record-video` if you want training without mp4 capture."
        )

    class EpisodeSummaryCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.best_summary: dict[str, Any] | None = load_json(RL_TRAINING_BEST_SUMMARY_PATH)
            self.current_episode_trace: list[dict[str, Any]] = []
            self.current_episode_video_path: Path | None = None
            self.current_episode_index = 0

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            rewards = self.locals.get("rewards", [])
            dones = self.locals.get("dones", [])
            for idx, info in enumerate(infos):
                reward_value = 0.0
                if len(rewards) > idx:
                    reward_value = float(rewards[idx])

                raw_sensors = info.get("raw_sensors")
                if raw_sensors:
                    if record_video and self.current_episode_video_path is None:
                        self.current_episode_index += 1
                        self.current_episode_video_path = run_dir / (
                            f"episode_{self.current_episode_index:05d}.mp4"
                        )
                        screen_recorder.start(self.current_episode_video_path)
                    self.current_episode_trace.append(
                        {
                            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                            "timesteps": int(self.num_timesteps),
                            "reward": reward_value,
                            "reward_breakdown": info.get("reward_breakdown"),
                            "policy_action": info.get("policy_action"),
                            "torcs_action": info.get("torcs_action"),
                            "sensors": raw_sensors,
                            "lap_completed": info.get("lap_completed"),
                            "lap_time": info.get("lap_time"),
                            "lap_source": info.get("lap_source"),
                            "done": bool(dones[idx]) if len(dones) > idx else False,
                        }
                    )

                summary = info.get("episode_summary")
                if not summary:
                    continue

                record = dict(summary)
                record["timesteps"] = int(self.num_timesteps)
                record["timestamp"] = datetime.now().isoformat(timespec="seconds")
                with episodes_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")

                self.logger.record("torcs/episode_reward", float(summary["episode_reward"]))
                self.logger.record("torcs/dist_raced", float(summary["dist_raced"]))
                self.logger.record("torcs/max_speed", float(summary["max_speed"]))
                self.logger.record("torcs/mean_speed", float(summary["mean_speed"]))
                self.logger.record(
                    "torcs/lap_time",
                    float(summary["lap_time"]) if summary.get("lap_time") is not None else 0.0,
                )

                recorded_video_path: Path | None = None
                if record_video and self.current_episode_video_path is not None:
                    recorded_video_path = screen_recorder.stop(keep_file=True)

                if self.best_summary is None or summary_rank(summary) < summary_rank(self.best_summary):
                    self.best_summary = record
                    self.model.save(str(best_model_path))
                    RL_TRAINING_BEST_SUMMARY_PATH.write_text(
                        json.dumps(record, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                    write_jsonl(RL_TRAINING_BEST_TELEMETRY_PATH, self.current_episode_trace)
                    write_jsonl(best_episode_telemetry_path, self.current_episode_trace)
                    if (
                        record_video
                        and recorded_video_path is not None
                        and recorded_video_path.exists()
                        and summary.get("lap_completed")
                        and summary.get("lap_time") is not None
                    ):
                        shutil.copyfile(recorded_video_path, RL_TRAINING_BEST_VIDEO_PATH)
                        shutil.copyfile(recorded_video_path, best_episode_video_path)
                    RL_TRAINING_BEST_TIME_PATH.write_text(
                        "\n".join(
                            [
                                (
                                    f"lap_time_seconds={float(summary['lap_time']):.6f}"
                                    if summary.get("lap_time") is not None
                                    else "lap_time_seconds="
                                ),
                                f"timestamp={record['timestamp']}",
                                f"run_dir={run_dir}",
                                f"summary_path={RL_TRAINING_BEST_SUMMARY_PATH}",
                                f"telemetry_path={RL_TRAINING_BEST_TELEMETRY_PATH}",
                                f"video_path={RL_TRAINING_BEST_VIDEO_PATH}",
                                f"model_path={best_model_path}.zip",
                                "source=rl_training",
                            ]
                        )
                        + "\n",
                        encoding="utf-8",
                    )

                    if summary.get("lap_completed") and summary.get("lap_time") is not None:
                        existing_global_best = load_json(BEST_LAP_SUMMARY_PATH)
                        if existing_global_best is None or summary_rank(record) < summary_rank(existing_global_best):
                            BEST_LAP_SUMMARY_PATH.write_text(
                                json.dumps(record, indent=2, sort_keys=True),
                                encoding="utf-8",
                            )
                            write_jsonl(BEST_LAP_TELEMETRY_PATH, self.current_episode_trace)
                            if (
                                record_video
                                and recorded_video_path is not None
                                and recorded_video_path.exists()
                            ):
                                shutil.copyfile(recorded_video_path, BEST_LAP_VIDEO_PATH)
                            BEST_LAP_TIME_PATH.write_text(
                                "\n".join(
                                    [
                                        f"lap_time_seconds={float(summary['lap_time']):.6f}",
                                        f"timestamp={record['timestamp']}",
                                        f"run_dir={run_dir}",
                                        f"summary_path={BEST_LAP_SUMMARY_PATH}",
                                        f"telemetry_path={BEST_LAP_TELEMETRY_PATH}",
                                        f"video_path={BEST_LAP_VIDEO_PATH}",
                                        f"model_path={best_model_path}.zip",
                                        "source=rl_training",
                                    ]
                                )
                                + "\n",
                                encoding="utf-8",
                            )

                if (
                    record_video
                    and recorded_video_path is not None
                    and recorded_video_path.exists()
                ):
                    keep_video = (
                        summary.get("lap_completed")
                        and summary.get("lap_time") is not None
                        and (
                            recorded_video_path == RL_TRAINING_BEST_VIDEO_PATH
                            or recorded_video_path == BEST_LAP_VIDEO_PATH
                        )
                    )
                    if not keep_video:
                        recorded_video_path.unlink(missing_ok=True)

                self.current_episode_video_path = None
                self.current_episode_trace = []
            return True

    env = Monitor(TorcsRLEnv(config), filename=str(monitor_path))

    model_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate_schedule,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=1,
        gradient_steps=1,
        learning_starts=args.learning_starts,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=str(TB_ROOT),
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    if args.resume_from:
        effective_resume_path = Path(args.resume_from)
    elif not args.fresh_start and latest_model_zip_path.exists():
        effective_resume_path = latest_model_zip_path

    if effective_resume_path is not None:
        print(f"Resuming SAC from: {effective_resume_path}")
        model = SAC.load(str(effective_resume_path), env=env, device=args.device)
        apply_learning_rate_to_model(model, learning_rate_schedule)
        effective_replay_buffer_path = (
            infer_replay_buffer_path(effective_resume_path)
            if args.resume_from
            else LATEST_REPLAY_BUFFER_PATH
        )
        if effective_replay_buffer_path.exists():
            try:
                model.load_replay_buffer(str(effective_replay_buffer_path))
                print(f"Loaded replay buffer: {effective_replay_buffer_path}")
            except Exception as exc:
                print(f"Could not load replay buffer from {effective_replay_buffer_path}: {exc}")
        else:
            print("No replay buffer found for resume; continuing from weights only.")
        reset_num_timesteps = False
    else:
        model = SAC(**model_kwargs)
        reset_num_timesteps = True

    episode_callback = EpisodeSummaryCallback()
    callbacks = CallbackList(
        [
            episode_callback,
            CheckpointCallback(
                save_freq=max(1, args.checkpoint_freq),
                save_path=str(MODEL_ROOT),
                name_prefix="sac_corkscrew_checkpoint",
                save_replay_buffer=True,
            ),
        ]
    )

    metadata = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "latest_model_path": str(latest_model_path),
        "best_model_path": str(best_model_path),
        "latest_replay_buffer_path": str(LATEST_REPLAY_BUFFER_PATH),
        "config": config.to_dict(),
        "args": vars(args),
        "effective_resume_path": str(effective_resume_path) if effective_resume_path is not None else "",
        "learning_rate_schedule": describe_learning_rate_schedule(args),
        "initial_learning_rate_value": resolve_learning_rate(learning_rate_schedule),
        "record_video": record_video,
        "best_episode_video_path": str(best_episode_video_path),
        "rl_training_best_video_path": str(RL_TRAINING_BEST_VIDEO_PATH),
        "best_lap_video_path": str(BEST_LAP_VIDEO_PATH),
    }
    training_summary_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("SAC training starting.")
    print(f"Run dir: {run_dir}")
    print(f"Learning-rate schedule: {describe_learning_rate_schedule(args)}")
    print(
        "Reminder: start `wtorcs.exe` first in `torcs/torcs`, click New Race, "
        "then launch this script from a second terminal in `torcs/gym_torcs`."
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name="sac_corkscrew",
            reset_num_timesteps=reset_num_timesteps,
        )
    finally:
        model.save(str(latest_model_path))
        try:
            model.save_replay_buffer(str(LATEST_REPLAY_BUFFER_PATH))
        except Exception as exc:
            print(f"Could not save replay buffer to {LATEST_REPLAY_BUFFER_PATH}: {exc}")
        if record_video and screen_recorder.is_recording():
            leftover_video = screen_recorder.stop(keep_file=False)
            if leftover_video is not None and leftover_video.exists():
                leftover_video.unlink(missing_ok=True)
        env.close()

    print(f"Latest SAC checkpoint saved to: {latest_model_path}.zip")
    print(f"Episode logs saved to: {episodes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
