from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from inference.config import InferenceConfig
from inference.exporter import export_bundle
from inference.runner import execute_inference
from rcss_env.action import Action
from train.train import build_rl_module_spec


class OneStepFakeEnv:
    def __init__(self, env_config: Any) -> None:
        self.agent_ids = env_config.curriculum.agent_unums()
        self.closed = False

    def _observations(self) -> dict[int, dict[str, np.ndarray]]:
        return {
            agent_id: {
                "obs": np.zeros((144,), dtype=np.float32),
                "action_mask": np.ones((Action.n_actions(),), dtype=np.int8),
            }
            for agent_id in self.agent_ids
        }

    def reset(self, *, seed=None, options=None):
        return self._observations(), {agent_id: {} for agent_id in self.agent_ids}

    def step(self, actions):
        assert set(actions) == set(self.agent_ids)
        return (
            self._observations(),
            {agent_id: 1.0 for agent_id in self.agent_ids},
            {**{agent_id: True for agent_id in self.agent_ids}, "__all__": True},
            {**{agent_id: False for agent_id in self.agent_ids}, "__all__": False},
            {
                agent_id: {"scores": {"our": 1, "their": 0}, "cycle": 7}
                for agent_id in self.agent_ids
            },
        )

    def close(self) -> None:
        self.closed = True


def _checkpoint_and_config(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "checkpoint_000001"
    build_rl_module_spec([2]).build().save_to_path(
        checkpoint / "learner_group" / "learner" / "rl_module"
    )
    train_config = tmp_path / "train.yaml"
    train_config.write_text(
        """
runtime:
  timestamp_experiment_name: false
  experiment_name: integration-test
curriculum:
  type: shooting
  agent_unum: 2
  team_side: left
  our_player_num: 2
  oppo_player_num: 1
logging:
  enable_aim: false
""".strip(),
        encoding="utf-8",
    )
    return checkpoint, train_config


def test_export_load_adapter_fake_env_episode(tmp_path: Path) -> None:
    checkpoint, train_config = _checkpoint_and_config(tmp_path)
    bundle = export_bundle(
        checkpoint_path=checkpoint,
        train_config_path=train_config,
        output_root=tmp_path / "models",
        model_name="integration-test",
        model_version="v1",
        experiment="integration-test",
        trial_id="trial-1",
        training_iteration=1,
        metric_name="checkpoint_score",
        metric_value=1.0,
        git_commit="abc123",
    )
    created: list[OneStepFakeEnv] = []

    def env_factory(env_config):
        env = OneStepFakeEnv(env_config)
        created.append(env)
        return env

    summary = execute_inference(
        InferenceConfig.model_validate(
            {
                "bundle": bundle,
                "device": "cpu",
                "episodes": 1,
                "curriculum": {"type": "shooting", "agent_unum": 2},
            }
        ),
        env_factory=env_factory,
    )

    assert len(summary.episodes) == 1
    assert summary.episodes[0].termination_reason == "terminated"
    assert summary.episodes[0].final_cycle == 7
    assert summary.episodes[0].decision_latency_summary["count"] == 1
    assert created[0].closed is True


def test_eleven_independent_modules_complete_one_fake_decision(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_000011"
    build_rl_module_spec(range(1, 12)).build().save_to_path(
        checkpoint / "learner_group" / "learner" / "rl_module"
    )
    train_config = tmp_path / "dummy.yaml"
    train_config.write_text(
        """
runtime:
  timestamp_experiment_name: false
  experiment_name: dummy-integration
curriculum:
  type: dummy_marl
  team_side: left
logging:
  enable_aim: false
""".strip(),
        encoding="utf-8",
    )
    bundle = export_bundle(
        checkpoint_path=checkpoint,
        train_config_path=train_config,
        output_root=tmp_path / "models",
        model_name="dummy-integration",
        model_version="v1",
        experiment="dummy-integration",
        trial_id="trial-11",
        training_iteration=1,
        metric_name="checkpoint_score",
        metric_value=1.0,
        git_commit="abc123",
    )
    created: list[OneStepFakeEnv] = []

    def env_factory(env_config):
        env = OneStepFakeEnv(env_config)
        created.append(env)
        return env

    summary = execute_inference(
        InferenceConfig.model_validate(
            {
                "bundle": bundle,
                "device": "cpu",
                "curriculum": {"type": "dummy_marl"},
            }
        ),
        env_factory=env_factory,
    )

    episode = summary.episodes[0]
    assert set(episode.rewards) == set(range(1, 12))
    assert set(episode.action_counts) == set(range(1, 12))
    assert created[0].closed is True
