from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from inference.exporter import export_bundle, main as exporter_main
from inference.loader import (
    BundleValidationError,
    ModelLoadError,
    load_bundle,
    validate_curriculum_agents,
)
from train.train import build_rl_module_spec, policy_id_for_agent


def _algorithm_checkpoint(tmp_path: Path, agent_ids: list[int]) -> Path:
    checkpoint = tmp_path / "checkpoint_000001"
    module_path = checkpoint / "learner_group" / "learner" / "rl_module"
    build_rl_module_spec(agent_ids).build().save_to_path(module_path)
    return checkpoint


def _shooting_config(tmp_path: Path, agent_unum: int = 2) -> Path:
    path = tmp_path / "shooting.yaml"
    path.write_text(
        f"""
runtime:
  timestamp_experiment_name: false
  experiment_name: export-test
curriculum:
  type: shooting
  agent_unum: {agent_unum}
  team_side: left
  our_player_num: 2
  oppo_player_num: 1
logging:
  enable_aim: false
""".strip(),
        encoding="utf-8",
    )
    return path


def _export(tmp_path: Path) -> Path:
    return export_bundle(
        checkpoint_path=_algorithm_checkpoint(tmp_path, [2]),
        train_config_path=_shooting_config(tmp_path),
        output_root=tmp_path / "models",
        model_name="shooting-test",
        model_version="v1",
        experiment="export-test",
        trial_id="trial-1",
        training_iteration=1,
        metric_name="checkpoint_score",
        metric_value=1.25,
        git_commit="abc123",
    )


def test_export_and_load_bundle_round_trip(tmp_path) -> None:
    bundle_path = _export(tmp_path)

    loaded = load_bundle(bundle_path, device="cpu")

    assert (bundle_path / "manifest.json").is_file()
    assert (bundle_path / "experiment.yaml").is_file()
    assert loaded.device == torch.device("cpu")
    assert loaded.manifest.agent_ids == (2,)
    assert set(loaded.module.keys()) == {policy_id_for_agent(2)}
    validate_curriculum_agents(loaded.manifest, [2])


def test_export_rejects_checkpoint_curriculum_topology_mismatch(tmp_path) -> None:
    checkpoint = _algorithm_checkpoint(tmp_path, [1, 2])

    with pytest.raises(BundleValidationError, match="module ids mismatch"):
        export_bundle(
            checkpoint_path=checkpoint,
            train_config_path=_shooting_config(tmp_path, agent_unum=2),
            output_root=tmp_path / "models",
            model_name="bad-topology",
            model_version="v1",
            experiment="export-test",
            trial_id="trial-1",
            training_iteration=1,
            metric_name="checkpoint_score",
            metric_value=0.0,
            git_commit="abc123",
        )


def test_export_refuses_to_overwrite_version(tmp_path) -> None:
    _export(tmp_path)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _export(tmp_path)


def test_validate_curriculum_agents_rejects_mismatch(tmp_path) -> None:
    loaded = load_bundle(_export(tmp_path), device="cpu")

    with pytest.raises(BundleValidationError, match="Curriculum agent ids mismatch"):
        validate_curriculum_agents(loaded.manifest, [1])


def test_loader_rejects_missing_manifest(tmp_path) -> None:
    bundle_path = _export(tmp_path)
    (bundle_path / "manifest.json").unlink()

    with pytest.raises(BundleValidationError, match="metadata"):
        load_bundle(bundle_path, device="cpu")


def test_loader_classifies_corrupt_model_as_model_error(tmp_path) -> None:
    bundle_path = _export(tmp_path)
    state_path = (
        bundle_path
        / "multi_rl_module"
        / policy_id_for_agent(2)
        / "module_state.pkl"
    )
    state_path.write_bytes(b"not-a-pickle")
    with pytest.raises(ModelLoadError, match="restore"):
        load_bundle(bundle_path, device="cpu")


def test_exporter_cli_consumes_best_checkpoint_metadata(tmp_path, capsys) -> None:
    checkpoint = _algorithm_checkpoint(tmp_path, [2])
    metadata = tmp_path / "best_checkpoint.json"
    metadata.write_text(
        json.dumps(
            {
                "experiment": "best-experiment",
                "trial_id": "trial-best",
                "checkpoint_uri": str(checkpoint),
                "metric": "checkpoint_score",
                "metric_value": 8.5,
                "metric_mode": "max",
                "training_iteration": 9,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    code = exporter_main(
        [
            "--best-checkpoint-json",
            str(metadata),
            "--train-config",
            str(_shooting_config(tmp_path)),
            "--output",
            str(tmp_path / "models"),
            "--model-name",
            "best-model",
            "--version",
            "v9",
            "--git-commit",
            "abc123",
        ]
    )

    assert code == 0
    output = Path(capsys.readouterr().out.strip())
    loaded = load_bundle(output, device="cpu")
    assert loaded.manifest.source.checkpoint_uri == checkpoint.resolve().as_posix()
    assert loaded.manifest.source.experiment == "best-experiment"
    assert loaded.manifest.source.trial_id == "trial-best"
    assert loaded.manifest.source.training_iteration == 9
    assert loaded.manifest.source.metric_value == pytest.approx(8.5)
