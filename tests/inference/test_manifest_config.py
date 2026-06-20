from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from inference.config import InferenceConfig, load_inference_config
from inference.manifest import BundleManifest


def _manifest_payload() -> dict:
    return {
        "schema_version": 1,
        "model_name": "dummy-marl-11v11",
        "model_version": "20260620.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "checkpoint_uri": "/tmp/checkpoint_000001",
            "experiment": "dummy-marl",
            "trial_id": "trial-1",
            "training_iteration": 1,
            "metric": "checkpoint_score",
            "metric_value": 1.0,
            "metric_mode": "max",
            "git_commit": "abc123",
        },
        "model": {
            "algorithm": "PPO",
            "module_class": "train.models.fcnet.RCSSPPOTorchRLModule",
            "module_ids": ["rcss_policy_1", "rcss_policy_2"],
            "stateful": False,
        },
        "policy_topology": {
            "mode": "independent_per_unum",
            "agent_to_module": {
                "1": "rcss_policy_1",
                "2": "rcss_policy_2",
            },
        },
        "observation": {
            "abi_version": "rcss-obs-v1",
            "shape": [144],
            "dtype": "float32",
        },
        "action": {
            "abi_version": "rcss-action-v1",
            "names": ["catch", "dash", "kick", "tackle", "turn"],
            "params_dim": 7,
            "params_dtype": "float32",
            "params_low": -1.0,
            "params_high": 1.0,
        },
        "inference": {
            "default_mode": "deterministic",
            "continuous_postprocess": "clip",
        },
    }


def test_manifest_accepts_independent_topology_round_trip(tmp_path) -> None:
    manifest = BundleManifest.model_validate(_manifest_payload())
    path = tmp_path / "manifest.json"
    path.write_text(manifest.to_json(), encoding="utf-8")

    restored = BundleManifest.model_validate_json(path.read_text(encoding="utf-8"))

    assert restored.agent_ids == (1, 2)
    assert set(restored.model.module_ids) == {"rcss_policy_1", "rcss_policy_2"}


@pytest.mark.parametrize(
    ("agent_id", "module_id"),
    [
        ("01", "rcss_policy_1"),
        ("0", "rcss_policy_0"),
        ("left", "rcss_policy_1"),
        ("1", "rcss_policy_2"),
    ],
)
def test_manifest_rejects_invalid_agent_mapping(
    agent_id: str,
    module_id: str,
) -> None:
    payload = _manifest_payload()
    payload["model"]["module_ids"] = [module_id]
    payload["policy_topology"]["agent_to_module"] = {agent_id: module_id}

    with pytest.raises(ValidationError):
        BundleManifest.model_validate(payload)


def test_manifest_rejects_module_set_mismatch() -> None:
    payload = _manifest_payload()
    payload["model"]["module_ids"] = ["rcss_policy_1"]

    with pytest.raises(ValidationError, match="exactly match"):
        BundleManifest.model_validate(payload)


def test_manifest_rejects_action_abi_drift() -> None:
    payload = _manifest_payload()
    payload["action"]["names"] = ["dash", "catch", "kick", "tackle", "turn"]

    with pytest.raises(ValidationError, match="Action names/order"):
        BundleManifest.model_validate(payload)


def test_inference_config_builds_dummy_marl_env_without_mapping() -> None:
    config = InferenceConfig.model_validate(
        {
            "bundle": "/tmp/model-bundle",
            "device": "cpu",
            "episodes": 2,
            "infrastructure": {
                "grpc": {"host": "127.0.0.1", "port": 0},
                "allocator": {"host": "allocator.test", "port": 8080},
            },
            "curriculum": {
                "type": "dummy_marl",
                "team_side": "left",
            },
        }
    )

    env_config = config.build_env_config()

    assert config.bundle_path == "/tmp/model-bundle"
    assert config.episodes == 2
    assert config.fallback_on_policy_error is True
    assert env_config.curriculum.agent_unums() == tuple(range(1, 12))
    assert env_config.allocator.base_url == "http://allocator.test:8080"


def test_inference_config_rejects_mapping_override() -> None:
    with pytest.raises(ValidationError, match="cannot override"):
        InferenceConfig.model_validate(
            {
                "bundle": "/tmp/model-bundle",
                "curriculum": {"type": "shooting"},
                "agent_to_module": {"1": "rcss_policy_1"},
            }
        )


def test_load_inference_config_from_yaml(tmp_path) -> None:
    path = tmp_path / "inference.yaml"
    path.write_text(
        """
bundle: /tmp/model-bundle
device: cpu
episodes: 1
curriculum:
  type: shooting
  agent_unum: 2
""".strip(),
        encoding="utf-8",
    )

    config = load_inference_config(path)

    assert config.curriculum_config.agent_unum == 2
    assert config.build_env_config().curriculum.agent_unums() == (2,)


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/inference/shooting.yaml",
        "configs/inference/dummy_marl_11v11.yaml",
    ],
)
def test_repository_inference_examples_remain_valid(relative_path: str) -> None:
    config = load_inference_config(Path(relative_path))

    assert config.episodes >= 1
    assert config.build_env_config().curriculum.agent_unums()
