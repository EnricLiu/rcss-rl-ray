"""Unit tests for training configuration and algo builder."""

from __future__ import annotations

import dataclasses
import pytest

from rcss_rl.config import EnvConfig, PlayerConfig, TrainConfig
from rcss_rl.train import _parse_args, build_algo


class TestEnvConfig:
    def test_defaults(self) -> None:
        cfg = EnvConfig()
        assert cfg.num_left == 3
        assert cfg.num_right == 3
        assert cfg.max_episode_steps == 200
        assert cfg.seed is None

    def test_num_left_right_derived(self) -> None:
        cfg = EnvConfig(
            ally_players=[
                PlayerConfig(unum=i, goalie=(i == 1))
                for i in range(1, 6)
            ],
            opponent_players=[
                PlayerConfig(unum=i, goalie=(i == 1))
                for i in range(1, 3)
            ],
        )
        assert cfg.num_left == 5
        assert cfg.num_right == 2


class TestTrainConfig:
    def test_defaults(self) -> None:
        cfg = TrainConfig()
        assert cfg.algo == "PPO"
        assert cfg.num_iterations == 100
        assert cfg.lr == pytest.approx(3e-4)

    def test_nested_env_config(self) -> None:
        cfg = TrainConfig()
        assert isinstance(cfg.env_config, EnvConfig)


class TestParseArgs:
    def test_defaults(self) -> None:
        cfg = _parse_args([])
        assert cfg.algo == "PPO"
        assert cfg.num_iterations == 100
        assert cfg.env_config.num_left == 3

    def test_custom_args(self) -> None:
        cfg = _parse_args(
            [
                "--algo", "IMPALA",
                "--iterations", "5",
                "--num-left", "2",
                "--num-right", "2",
                "--seed", "42",
            ]
        )
        assert cfg.algo == "IMPALA"
        assert cfg.num_iterations == 5
        assert cfg.env_config.num_left == 2
        assert cfg.env_config.seed == 42

    def test_invalid_algo_raises(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args(["--algo", "UNKNOWN_ALGO"])


class TestBuildAlgo:
    def test_invalid_algo_raises_value_error(self) -> None:
        cfg = TrainConfig(algo="FAKE_ALGO")
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            build_algo(cfg)
