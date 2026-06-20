from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import signal
from typing import Any

import numpy as np
import pytest
import torch

from inference import cli
from inference.config import InferenceConfig
from inference.loader import BundleValidationError, LoadedBundle, ModelLoadError
from inference.manifest import BundleManifest
from inference.policy import PolicyInferenceError
from inference.runner import (
    InferenceRunner,
    RunSummary,
    RunnerInfrastructureError,
    execute_inference,
)
from train.train import build_rl_module_spec, policy_id_for_agent


class FakePolicy:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error

    def compute_actions(self, observations):
        if self.error is not None:
            raise self.error
        return {
            agent_id: {
                "actions": 1,
                "params": np.zeros((7,), dtype=np.float32),
            }
            for agent_id in observations
        }


@dataclass
class FakeEnv:
    reset_needed_once: bool = False
    reset_error_once: bool = False

    def __post_init__(self) -> None:
        self.closed = False
        self.reset_calls = 0
        self.step_calls = 0
        self.episode_step = 0

    def reset(self, *, seed=None, options=None):
        self.reset_calls += 1
        self.episode_step = 0
        if self.reset_error_once and self.reset_calls == 1:
            raise RuntimeError("reset transport failed")
        return {2: {"obs": object()}}, {2: {}}

    def step(self, actions: dict[int, Any]):
        self.step_calls += 1
        self.episode_step += 1
        obs = {2: {"obs": object()}}
        if self.reset_needed_once and self.reset_calls == 1:
            return (
                obs,
                {2: 0.0},
                {2: False, "__all__": False},
                {2: True, "__all__": True},
                {2: {"reset_needed": True}},
            )
        done = self.episode_step >= 2
        return (
            obs,
            {2: 0.5},
            {2: done, "__all__": done},
            {2: False, "__all__": False},
            {2: {"scores": {"our": 1, "their": 0}, "cycle": 42}},
        )

    def close(self) -> None:
        self.closed = True


class MalformedMetricsEnv(FakeEnv):
    def step(self, actions: dict[int, Any]):
        self.step_calls += 1
        return (
            {2: {"obs": object()}},
            {2: "not-a-reward"},
            {2: True, "__all__": True},
            {2: False, "__all__": False},
            {2: {"scores": {"our": "bad", "their": 0}, "cycle": "bad"}},
        )


class CloseErrorEnv(FakeEnv):
    def close(self) -> None:
        self.closed = True
        raise RuntimeError("cleanup failed")


def test_runner_completes_episodes_and_closes_env() -> None:
    env = FakeEnv()
    runner = InferenceRunner(env=env, policy=FakePolicy(), episodes=2, seed=10)

    summary = runner.run()

    assert env.closed is True
    assert len(summary.episodes) == 2
    assert summary.episodes[0].steps == 2
    assert summary.episodes[0].rewards == {2: pytest.approx(1.0)}
    assert summary.episodes[0].our_score == 1.0
    assert summary.episodes[0].final_cycle == 42
    assert summary.episodes[0].episode_seconds >= 0.0
    assert summary.episodes[0].decision_latency_summary["count"] == 2
    assert summary.episodes[0].action_counts == {2: {"dash": 2}}
    assert summary.to_dict()["episodes"][0]["decision_latency_seconds"]["p99"] >= 0.0
    assert summary.metrics["episodes_requested"] == 2
    assert summary.metrics["episodes_completed"] == 2
    assert summary.metrics["episode_steps_total"] == 4
    assert summary.metrics["actions_total"] == {"dash": 4}
    assert summary.metrics["decision_latency_seconds"]["count"] == 4
    assert summary.episodes[0].termination_reason == "terminated"


def test_runner_retries_reset_needed_with_new_reset() -> None:
    env = FakeEnv(reset_needed_once=True)
    runner = InferenceRunner(
        env=env,
        policy=FakePolicy(),
        episodes=1,
        max_episode_retries=1,
    )

    summary = runner.run()

    assert env.reset_calls == 2
    assert summary.episodes[0].attempts == 2
    assert env.closed is True


def test_runner_retries_reset_exception() -> None:
    env = FakeEnv(reset_error_once=True)
    runner = InferenceRunner(
        env=env,
        policy=FakePolicy(),
        episodes=1,
        max_episode_retries=1,
    )

    summary = runner.run()

    assert env.reset_calls == 2
    assert summary.episodes[0].attempts == 2
    assert env.closed is True


def test_runner_exhausted_infrastructure_retry_closes_env() -> None:
    env = FakeEnv(reset_needed_once=True)
    runner = InferenceRunner(env=env, policy=FakePolicy(), episodes=1)

    with pytest.raises(RunnerInfrastructureError, match="failed after"):
        runner.run()

    assert env.closed is True


def test_runner_policy_error_is_not_retried_and_closes_env() -> None:
    env = FakeEnv()
    error = ValueError("model failed")
    runner = InferenceRunner(env=env, policy=FakePolicy(error=error), episodes=1)

    with pytest.raises(ValueError, match="model failed"):
        runner.run()

    assert env.reset_calls == 1
    assert env.closed is True


def test_runner_ignores_malformed_metrics(caplog) -> None:
    env = MalformedMetricsEnv()
    with caplog.at_level(logging.WARNING, logger="inference.runner"):
        summary = InferenceRunner(env=env, policy=FakePolicy(), episodes=1).run()

    episode = summary.episodes[0]
    assert episode.termination_reason == "terminated"
    assert episode.rewards == {2: 0.0}
    assert episode.our_score is None
    assert episode.final_cycle is None
    assert "Ignoring malformed reward" in caplog.text
    assert "Ignoring malformed score metadata" in caplog.text
    assert "Ignoring malformed cycle metadata" in caplog.text


def test_runner_logs_close_failure_without_losing_result(caplog) -> None:
    env = CloseErrorEnv()
    with caplog.at_level(logging.ERROR, logger="inference.runner"):
        summary = InferenceRunner(env=env, policy=FakePolicy(), episodes=1).run()

    assert summary.episodes[0].termination_reason == "terminated"
    assert env.closed is True
    assert "Failed to close inference environment" in caplog.text


def test_runner_stop_request_closes_without_reset() -> None:
    env = FakeEnv()
    runner = InferenceRunner(
        env=env,
        policy=FakePolicy(),
        episodes=1,
        stop_requested=lambda: True,
    )

    summary = runner.run()

    assert summary.interrupted is True
    assert env.reset_calls == 0
    assert env.closed is True


def test_runner_marks_mid_episode_stop_as_interrupted() -> None:
    env = FakeEnv()
    checks = {"count": 0}

    def stop_requested() -> bool:
        checks["count"] += 1
        return checks["count"] >= 4

    summary = InferenceRunner(
        env=env,
        policy=FakePolicy(),
        episodes=1,
        stop_requested=stop_requested,
    ).run()

    assert summary.interrupted is True
    assert summary.episodes[0].termination_reason == "interrupted"
    assert env.closed is True


def _loaded_bundle_for_agent(agent_id: int, tmp_path: Path) -> LoadedBundle:
    module_id = policy_id_for_agent(agent_id)
    manifest = BundleManifest.model_validate(
        {
            "model_name": "runner-test",
            "model_version": "v1",
            "created_at": datetime.now(timezone.utc),
            "source": {
                "checkpoint_uri": "/tmp/checkpoint",
                "experiment": "test",
                "trial_id": "trial",
                "training_iteration": 1,
                "metric": "checkpoint_score",
                "metric_value": 0.0,
                "metric_mode": "max",
                "git_commit": "abc123",
            },
            "model": {"module_ids": [module_id]},
            "policy_topology": {
                "agent_to_module": {str(agent_id): module_id}
            },
        }
    )
    return LoadedBundle(
        path=tmp_path,
        manifest=manifest,
        module=build_rl_module_spec([agent_id]).build(),
        device=torch.device("cpu"),
    )


def test_execute_inference_checks_topology_before_env_construction(tmp_path) -> None:
    loaded = _loaded_bundle_for_agent(2, tmp_path)
    config = InferenceConfig.model_validate(
        {
            "bundle": tmp_path,
            "curriculum": {"type": "shooting", "agent_unum": 1},
        }
    )
    env_constructed = {"value": False}

    def env_factory(env_config):
        env_constructed["value"] = True
        raise AssertionError("Environment must not be constructed")

    with pytest.raises(BundleValidationError, match="Curriculum agent ids mismatch"):
        execute_inference(
            config,
            env_factory=env_factory,
            bundle_loader=lambda *args, **kwargs: loaded,
        )

    assert env_constructed["value"] is False


def _cli_config(tmp_path: Path) -> Path:
    path = tmp_path / "inference.yaml"
    path.write_text(
        f"""
bundle: {tmp_path / 'bundle'}
device: cpu
episodes: 1
curriculum:
  type: shooting
  agent_unum: 2
""".strip(),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (BundleValidationError("bad manifest"), 2),
        (RunnerInfrastructureError("allocator failed"), 3),
        (ModelLoadError("restore failed"), 4),
        (PolicyInferenceError("forward failed"), 4),
    ],
)
def test_cli_stable_error_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_code: int,
) -> None:
    monkeypatch.setattr(cli, "execute_inference", lambda *args, **kwargs: (_ for _ in ()).throw(error))

    assert cli.main(["--config", str(_cli_config(tmp_path))]) == expected_code


def test_cli_revalidates_overrides(tmp_path: Path) -> None:
    assert cli.main(
        ["--config", str(_cli_config(tmp_path)), "--episodes", "0"]
    ) == 2


def test_cli_success_and_signal_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_logger = logging.getLogger()
    previous_handlers = list(root_logger.handlers)
    previous_level = root_logger.level
    monkeypatch.setattr(cli, "execute_inference", lambda *args, **kwargs: RunSummary(()))
    config_path = _cli_config(tmp_path)
    assert cli.main(["--config", str(config_path)]) == 0

    def interrupt(config, *, stop_requested):
        signal.raise_signal(signal.SIGINT)
        assert stop_requested() is True
        return RunSummary((), interrupted=True)

    monkeypatch.setattr(cli, "execute_inference", interrupt)
    assert cli.main(["--config", str(config_path)]) == 130
    assert root_logger.handlers == previous_handlers
    assert root_logger.level == previous_level
