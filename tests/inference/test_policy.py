from __future__ import annotations

from datetime import datetime, timezone
import logging

import numpy as np
import pytest
import torch

from inference.manifest import BundleManifest
from inference.policy import (
    MultiAgentPolicyAdapter,
    PolicyInferenceError,
    decode_action_sample,
    fallback_action,
    seed_inference,
)
from rcss_env.action import Action
from train.train import build_rl_module_spec, policy_id_for_agent


def _manifest(agent_ids: list[int]) -> BundleManifest:
    module_ids = [policy_id_for_agent(agent_id) for agent_id in agent_ids]
    return BundleManifest.model_validate(
        {
            "schema_version": 1,
            "model_name": "policy-test",
            "model_version": "v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
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
            "model": {"module_ids": module_ids},
            "policy_topology": {
                "agent_to_module": {
                    str(agent_id): policy_id_for_agent(agent_id)
                    for agent_id in agent_ids
                }
            },
        }
    )


def _observations(agent_ids: list[int]) -> dict[int, dict[str, np.ndarray]]:
    return {
        agent_id: {
            "obs": np.zeros((144,), dtype=np.float32),
            "action_mask": np.ones((Action.n_actions(),), dtype=np.int8),
        }
        for agent_id in reversed(agent_ids)
    }


def _adapter(agent_ids: list[int], deterministic: bool = True) -> MultiAgentPolicyAdapter:
    module = build_rl_module_spec(agent_ids).build()
    for module_id in module.keys():
        module[module_id].eval()
    return MultiAgentPolicyAdapter(
        module=module,
        manifest=_manifest(agent_ids),
        device=torch.device("cpu"),
        deterministic=deterministic,
    )


def test_deterministic_adapter_handles_unordered_non_contiguous_agents() -> None:
    adapter = _adapter([2, 7])
    observations = _observations([2, 7])

    first = adapter.compute_actions(observations)
    second = adapter.compute_actions(observations)

    assert set(first) == {2, 7}
    for agent_id in first:
        assert first[agent_id]["actions"] == second[agent_id]["actions"]
        np.testing.assert_array_equal(
            first[agent_id]["params"],
            second[agent_id]["params"],
        )
        assert first[agent_id]["params"].dtype == np.float32
        assert first[agent_id]["params"].shape == (Action.n_action_params(),)
        assert np.abs(first[agent_id]["params"]).max() <= 1.0


def test_action_mask_can_force_each_agent_to_one_action() -> None:
    adapter = _adapter([2, 7])
    observations = _observations([2, 7])
    observations[2]["action_mask"][:] = 0
    observations[2]["action_mask"][Action.action_index(Action.DASH)] = 1
    observations[7]["action_mask"][:] = 0
    observations[7]["action_mask"][Action.action_index(Action.TURN)] = 1

    actions = adapter.compute_actions(observations)

    assert actions[2]["actions"] == Action.action_index(Action.DASH)
    assert actions[7]["actions"] == Action.action_index(Action.TURN)


@pytest.mark.parametrize("actual_ids", [[2], [2, 7, 9]])
def test_adapter_rejects_agent_set_mismatch(actual_ids: list[int]) -> None:
    adapter = _adapter([2, 7])

    with pytest.raises(PolicyInferenceError, match="agent set mismatch"):
        adapter.compute_actions(_observations(actual_ids))


def test_adapter_rejects_invalid_observation_and_mask() -> None:
    adapter = _adapter([2])
    observations = _observations([2])
    observations[2]["obs"] = np.zeros((144,), dtype=np.float64)
    with pytest.raises(PolicyInferenceError, match="float32"):
        adapter.compute_actions(observations)

    observations = _observations([2])
    observations[2]["action_mask"][:] = 0
    with pytest.raises(PolicyInferenceError, match="no legal action"):
        adapter.compute_actions(observations)


def test_decode_action_clips_params_and_rejects_masked_action() -> None:
    mask = np.ones((Action.n_actions(),), dtype=np.int8)
    sample = {
        "actions": torch.tensor([Action.action_index(Action.DASH)]),
        "params": torch.tensor([[2.0] * Action.n_action_params()]),
    }

    decoded = decode_action_sample(agent_id=2, sample=sample, action_mask=mask)

    np.testing.assert_array_equal(
        decoded["params"],
        np.ones((Action.n_action_params(),), dtype=np.float32),
    )

    mask[Action.action_index(Action.DASH)] = 0
    with pytest.raises(PolicyInferenceError, match="masked action"):
        decode_action_sample(agent_id=2, sample=sample, action_mask=mask)


def test_stochastic_adapter_is_seedable() -> None:
    adapter = _adapter([2], deterministic=False)
    observations = _observations([2])

    seed_inference(123)
    first = adapter.compute_actions(observations)
    seed_inference(123)
    second = adapter.compute_actions(observations)

    assert first[2]["actions"] == second[2]["actions"]
    np.testing.assert_array_equal(first[2]["params"], second[2]["params"])


def test_adapter_wraps_forward_failure_as_policy_error(monkeypatch) -> None:
    adapter = _adapter([2])

    def fail_forward(batch):
        raise RuntimeError("torch failed")

    monkeypatch.setattr(adapter.module, "forward_inference", fail_forward)

    with pytest.raises(PolicyInferenceError, match="forward_inference failed"):
        adapter.compute_actions(_observations([2]))


def test_fallback_action_prefers_zero_turn_and_respects_mask() -> None:
    action = fallback_action()
    assert action["actions"] == Action.action_index(Action.TURN)
    assert action["params"][6] == pytest.approx(0.0)

    mask = Action.mask_from_allowed([Action.DASH])
    action = fallback_action(mask)
    assert action["actions"] == Action.action_index(Action.DASH)
    assert action["params"][0] == pytest.approx(-1.0)


def test_resilient_adapter_falls_back_only_for_invalid_agent(caplog) -> None:
    adapter = _adapter([2, 7])
    adapter.fallback_on_error = True
    observations = _observations([2, 7])
    observations[2]["obs"] = np.zeros((144,), dtype=np.float64)

    with caplog.at_level(logging.WARNING, logger="inference.policy"):
        actions = adapter.compute_actions(observations)
        adapter.compute_actions(observations)

    assert set(actions) == {2, 7}
    assert actions[2]["actions"] == Action.action_index(Action.TURN)
    assert "Agent 2 using fallback action" in caplog.text
    assert "Agent 7 using fallback action" not in caplog.text
    fallback_warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "Agent 2 using fallback action" in record.getMessage()
    ]
    assert len(fallback_warnings) == 1

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="inference.policy"):
        adapter.compute_actions(_observations([2, 7]))
    assert "recovered after 2 fallback action(s)" in caplog.text


def test_resilient_adapter_uses_one_error_log_for_forward_failure(
    monkeypatch,
    caplog,
) -> None:
    adapter = _adapter([2, 7])
    adapter.fallback_on_error = True

    def fail_forward(batch):
        raise RuntimeError("device hiccup")

    monkeypatch.setattr(adapter.module, "forward_inference", fail_forward)
    with caplog.at_level(logging.WARNING, logger="inference.policy"):
        actions = adapter.compute_actions(_observations([2, 7]))

    assert all(
        action["actions"] == Action.action_index(Action.TURN)
        for action in actions.values()
    )
    error_records = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert "all affected agents" in error_records[0].getMessage()
