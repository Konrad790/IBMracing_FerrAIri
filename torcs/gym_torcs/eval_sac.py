from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "models" / "sac_corkscrew"
RESULTS_ROOT = REPO_ROOT / "results" / "rl_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SAC policy in TORCS."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_ROOT / "sac_corkscrew_latest.zip"),
        help="Path to a trained SAC checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes.")
    parser.add_argument("--port", type=int, default=3001, help="SCR UDP port used by TORCS.")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=9000,
        help="Maximum number of steps per evaluation episode.",
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
    return parser.parse_args()


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

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model file does not exist: {model_path}")

    run_dir = RESULTS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = run_dir / "episodes.jsonl"

    env = TorcsRLEnv(
        TorcsEnvConfig(port=args.port, max_episode_steps=args.max_episode_steps)
    )
    model = SAC.load(str(model_path), device=args.device)

    print(f"Evaluating model: {model_path}")
    print(f"Run dir: {run_dir}")
    print(
        "Reminder: start `wtorcs.exe` first in `torcs/torcs`, click New Race, "
        "then launch this script from a second terminal in `torcs/gym_torcs`."
    )

    try:
        for episode_index in range(1, args.episodes + 1):
            observation, _info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            final_info: dict[str, object] = {}

            while not (terminated or truncated):
                action, _state = model.predict(
                    observation,
                    deterministic=not args.stochastic,
                )
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                final_info = info

            summary = dict(final_info.get("episode_summary", {}))
            summary["episode_index"] = episode_index
            summary["policy_mode"] = "stochastic" if args.stochastic else "deterministic"
            summary["total_reward"] = total_reward
            summary["timestamp"] = datetime.now().isoformat(timespec="seconds")

            with episodes_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, sort_keys=True) + "\n")

            lap_time = summary.get("lap_time")
            lap_text = f"{lap_time:.3f}s" if isinstance(lap_time, (int, float)) else "n/a"
            print(
                f"Episode {episode_index}: lap_completed={summary.get('lap_completed')}, "
                f"lap_time={lap_text}, dist_raced={summary.get('dist_raced')}, "
                f"reason={summary.get('termination_reason')}"
            )
    finally:
        env.close()

    print(f"Evaluation logs saved to: {episodes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
