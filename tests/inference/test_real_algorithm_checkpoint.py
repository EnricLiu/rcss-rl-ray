from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import os

import numpy as np
import pytest
import ray
from gymnasium import spaces
from ray import tune
from ray.rllib.algorithms import algorithm as algorithm_module
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from inference.exporter import export_bundle
from inference.loader import load_bundle
from rcss_env import obs as observation
from rcss_env.action import Action
from train.train import (
    build_rl_module_spec,
    independent_policy_mapping_fn,
    policy_id_for_agent,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_RAY_ALGORITHM_CHECKPOINT_TEST") != "1",
    reason="requires a host that can start a local Ray runtime",
)


class CheckpointEnv(MultiAgentEnv):
    def __init__(self, config=None) -> None:
        super().__init__()
        self.possible_agents = [2]
        self.agents = [2]
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    -np.inf,
                    np.inf,
                    (observation.dim(),),
                    np.float32,
                ),
                "action_mask": spaces.Box(
                    0,
                    1,
                    (Action.n_actions(),),
                    np.int8,
                ),
            }
        )
        self.action_space = Action.space_schema()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.agents = [2]
        return {
            2: {
                "obs": np.zeros((observation.dim(),), dtype=np.float32),
                "action_mask": np.ones((Action.n_actions(),), dtype=np.int8),
            }
        }, {2: {}}

    def step(self, action_dict):
        self.agents = []
        return (
            {},
            {2: 0.0},
            {2: True, "__all__": True},
            {2: False, "__all__": False},
            {2: {}},
        )

    def get_observation_space(self, agent_id):
        return self.observation_space

    def get_action_space(self, agent_id):
        return self.action_space


def test_real_algorithm_checkpoint_exports_and_loads(tmp_path, monkeypatch) -> None:
    env_name = "inference_real_algorithm_checkpoint_env"
    tune.register_env(env_name, lambda config: CheckpointEnv(config))
    monkeypatch.setattr(
        algorithm_module,
        "DEFAULT_STORAGE_PATH",
        str(tmp_path / "ray-results"),
    )
    ray_runtime = Path(tempfile.mkdtemp(prefix="ray-", dir="/tmp"))
    algorithm = None
    try:
        ray.init(
            num_cpus=1,
            include_dashboard=False,
            ignore_reinit_error=True,
            _temp_dir=str(ray_runtime),
        )
        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .environment(env_name)
            .env_runners(num_env_runners=0)
            .learners(num_learners=0)
            .framework("torch")
            .rl_module(rl_module_spec=build_rl_module_spec([2]))
            .multi_agent(
                policies={policy_id_for_agent(2)},
                policy_mapping_fn=independent_policy_mapping_fn,
            )
        )
        algorithm = config.build_algo()
        checkpoint = tmp_path / "checkpoint_000001"
        algorithm.save_to_path(checkpoint)
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()
        shutil.rmtree(ray_runtime, ignore_errors=True)

    module_root = checkpoint / "learner_group" / "learner" / "rl_module"
    assert (module_root / "metadata.json").is_file()
    assert (module_root / policy_id_for_agent(2) / "module_state.pkl").is_file()

    train_config = tmp_path / "train.yaml"
    train_config.write_text(
        """
runtime:
  timestamp_experiment_name: false
  experiment_name: real-checkpoint-test
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
    bundle = export_bundle(
        checkpoint_path=checkpoint,
        train_config_path=train_config,
        output_root=tmp_path / "models",
        model_name="real-checkpoint-test",
        model_version="v1",
        experiment="real-checkpoint-test",
        trial_id="trial-1",
        training_iteration=0,
        metric_name="checkpoint_score",
        metric_value=0.0,
        git_commit="abc123",
    )

    loaded = load_bundle(bundle, device="cpu")
    assert loaded.manifest.agent_ids == (2,)
    assert set(loaded.module.keys()) == {policy_id_for_agent(2)}
