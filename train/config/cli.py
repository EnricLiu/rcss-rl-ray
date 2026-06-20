from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .loader import load_train_config
from .schema import TrainConfig


def _optional_int(value: str) -> int | None:
    if value.lower() in {"", "none", "null"}:
        return None
    return int(value)


def _float_or_auto(value: str) -> float | str:
    if value.lower() == "auto":
        return "auto"
    return float(value)


def _csv_tuple(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on RCSSEnv from a YAML, JSON, or TOML config file."
    )
    parser.add_argument("-f", "--config", dest="config_path", type=Path, default=None)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    # Small, common runtime overrides for config-file launches.
    parser.add_argument("--ray-address", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--experiment-name", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--storage-path", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--restore", dest="restore_path", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--resume-from-checkpoint", dest="resume_from_checkpoint", type=str, default=argparse.SUPPRESS)
    parser.add_argument(
        "--warm-start-module-checkpoint",
        dest="warm_start_module_checkpoint",
        type=str,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--disable-aim", dest="enable_aim", action="store_false", default=argparse.SUPPRESS)
    parser.add_argument("--aim-repo", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--aim-experiment-name", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--no-timestamp-experiment-name", dest="timestamp_experiment_name", action="store_false", default=argparse.SUPPRESS)

    # Legacy overrides kept for existing scripts/tests. They are hidden from
    # help so the public entry point stays config-file oriented.
    legacy = argparse.SUPPRESS
    parser.add_argument("--timestamp-experiment-name", dest="timestamp_experiment_name", action="store_true", default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--num-samples", type=int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--metric", type=str, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--checkpoint-metric", type=str, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--checkpoint-source-metric", type=str, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--mode", choices=["min", "max"], default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--log-to-file", action="store_true", default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--num-env-runners", type=int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--num-envs-per-runner", type=int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--num-cpus-per-runner", type=float, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--num-learners", type=int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--num-cpus-per-learner", type=_float_or_auto, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--num-gpus-per-learner", type=float, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--train-batch-size", type=int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--sgd-minibatch-size", type=int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--num-sgd-iter", type=int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--gamma", type=float, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--entropy-coeff", type=float, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--clip-param", type=float, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--num-iterations", type=int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--checkpoint-freq", type=int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--checkpoint-num-to-keep", type=_optional_int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--no-checkpoint-at-end", dest="checkpoint_at_end", action="store_false", default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--grpc-host", type=str, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--grpc-port", type=int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--allocator-host", type=str, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--allocator-port", type=int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--curriculum", choices=["shooting", "dummy_marl"], default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--curriculum-debug", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--agent-unum", type=int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--team-side", choices=["left", "right", "rand"], default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--our-player-num", type=int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--oppo-player-num", type=int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--our-goalie-unum", type=_optional_int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--oppo-goalie-unum", type=_optional_int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--our-team-name", type=str, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--oppo-team-name", type=str, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--agent-image", dest="player_agent_image", type=str, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--bot-image", dest="player_bot_image", type=str, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--time-up", type=int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--goal-l", type=_optional_int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--goal-r", type=_optional_int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--reward-goal", type=float, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--reward-concede", type=float, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--reward-out-of-bounds", type=float, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--reward-kickable-bonus", type=float, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--reward-agent-to-ball-shaping", type=float, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--reward-ball-to-goal-shaping", type=float, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--reward-ball-velocity-to-goal", type=float, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--gamma-shaping", type=float, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--shaping-clip", type=float, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--reward-time-decay", type=float, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--max-cycle-gap", type=int, default=argparse.SUPPRESS, help=legacy)
    parser.add_argument("--aim-metrics", type=str, default=argparse.SUPPRESS, help=legacy)

    return parser.parse_args(argv)


def _namespace_overrides(args: argparse.Namespace) -> dict[str, Any]:
    ignored = {"config_path", "log_level"}
    payload = {
        key: value
        for key, value in vars(args).items()
        if key not in ignored
    }
    if "aim_metrics" in payload:
        payload["aim_metrics"] = _csv_tuple(payload["aim_metrics"])
    return payload


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    overrides = _namespace_overrides(args)
    config_path = getattr(args, "config_path", None)
    if config_path is not None:
        return load_train_config(config_path, overrides=overrides)
    return TrainConfig.model_validate(overrides)
