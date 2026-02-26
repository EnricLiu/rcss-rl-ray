"""Unit tests for RCSSEnv."""

from __future__ import annotations

import numpy as np
import pytest

from rcss_rl.config import EnvConfig
from rcss_rl.env.rcss_env import RCSSEnv


@pytest.fixture()
def env() -> RCSSEnv:
    cfg = EnvConfig(num_left=2, num_right=2, max_episode_steps=50, seed=42)
    return RCSSEnv(cfg)


class TestRCSSEnvSpaces:
    def test_observation_space_shape(self, env: RCSSEnv) -> None:
        assert env.observation_space.shape == (18,)

    def test_action_space_size(self, env: RCSSEnv) -> None:
        assert env.action_space.n == 6

    def test_agent_ids(self, env: RCSSEnv) -> None:
        expected = {"left_0", "left_1", "right_0", "right_1"}
        assert env._agent_ids == expected


class TestRCSSEnvReset:
    def test_reset_returns_obs_for_all_agents(self, env: RCSSEnv) -> None:
        obs, infos = env.reset()
        assert set(obs.keys()) == env._agent_ids
        assert set(infos.keys()) == env._agent_ids

    def test_reset_obs_shape(self, env: RCSSEnv) -> None:
        obs, _ = env.reset()
        for agent_id, o in obs.items():
            assert o.shape == (18,), f"Wrong shape for {agent_id}"
            assert o.dtype == np.float32

    def test_reset_with_seed(self, env: RCSSEnv) -> None:
        obs1, _ = env.reset(seed=0)
        obs2, _ = env.reset(seed=0)
        for agent_id in env._agent_ids:
            np.testing.assert_array_equal(obs1[agent_id], obs2[agent_id])

    def test_reset_different_seeds_differ(self, env: RCSSEnv) -> None:
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=99)
        # At least one agent's obs should differ.
        any_differ = any(
            not np.array_equal(obs1[aid], obs2[aid]) for aid in env._agent_ids
        )
        assert any_differ


class TestRCSSEnvStep:
    def test_step_returns_correct_keys(self, env: RCSSEnv) -> None:
        env.reset()
        action_dict = {aid: env.action_space.sample() for aid in env._agent_ids}
        obs, rew, term, trunc, info = env.step(action_dict)
        assert set(obs.keys()) == env._agent_ids
        assert set(rew.keys()) == env._agent_ids
        assert "__all__" in term
        assert "__all__" in trunc

    def test_step_obs_dtype_and_shape(self, env: RCSSEnv) -> None:
        env.reset()
        action_dict = {aid: 0 for aid in env._agent_ids}
        obs, _, _, _, _ = env.step(action_dict)
        for agent_id, o in obs.items():
            assert o.shape == (18,), f"Bad shape for {agent_id}"
            assert o.dtype == np.float32

    def test_episode_terminates_after_max_steps(self, env: RCSSEnv) -> None:
        env.reset()
        done = False
        for _ in range(50):
            action_dict = {aid: env.action_space.sample() for aid in env._agent_ids}
            _, _, _, trunc, _ = env.step(action_dict)
            if trunc["__all__"]:
                done = True
                break
        assert done, "Episode should truncate after max_episode_steps=50"

    def test_step_count_increments(self, env: RCSSEnv) -> None:
        env.reset()
        assert env._step_count == 0
        env.step({aid: 0 for aid in env._agent_ids})
        assert env._step_count == 1

    def test_rewards_are_finite(self, env: RCSSEnv) -> None:
        env.reset(seed=7)
        for _ in range(10):
            action_dict = {aid: env.action_space.sample() for aid in env._agent_ids}
            _, rew, _, _, _ = env.step(action_dict)
            for agent_id, r in rew.items():
                assert np.isfinite(r), f"Non-finite reward for {agent_id}"

    def test_partial_action_dict(self, env: RCSSEnv) -> None:
        """Env should not crash if only a subset of agents provide actions."""
        env.reset()
        partial = {"left_0": 0, "right_0": 3}
        obs, rew, term, trunc, info = env.step(partial)
        # Only the acting agents appear in rewards.
        assert set(rew.keys()) == {"left_0", "right_0"}


class TestRCSSEnvConfig:
    def test_dict_config(self) -> None:
        env = RCSSEnv({"num_left": 1, "num_right": 1})
        assert len(env._agent_ids) == 2

    def test_default_config(self) -> None:
        env = RCSSEnv()
        assert env._cfg.num_left == 3
        assert env._cfg.num_right == 3
