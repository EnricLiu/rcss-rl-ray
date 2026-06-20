from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule

from inference.migrate import migrate_shared_checkpoint
from train.train import build_rl_module_spec, policy_id_for_agent


def _legacy_algorithm_checkpoint(tmp_path: Path) -> tuple[Path, dict]:
    checkpoint = tmp_path / "legacy-checkpoint"
    leaf = (
        checkpoint
        / "learner_group"
        / "learner"
        / "rl_module"
        / "rcss_policy"
    )
    source = build_rl_module_spec([1]).build()[policy_id_for_agent(1)]
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.125)
    source.save_to_path(leaf)
    return checkpoint, source.get_state()


def test_shared_checkpoint_migration_clones_independent_parameters(tmp_path) -> None:
    source, source_state = _legacy_algorithm_checkpoint(tmp_path)

    output = migrate_shared_checkpoint(
        source_checkpoint=source,
        output_path=tmp_path / "migration",
        target_agent_ids=[2, 7],
    )
    migrated = MultiRLModule.from_checkpoint(output / "multi_rl_module")

    assert set(migrated.keys()) == {policy_id_for_agent(2), policy_id_for_agent(7)}
    module_2 = migrated[policy_id_for_agent(2)]
    module_7 = migrated[policy_id_for_agent(7)]
    assert next(module_2.parameters()).data_ptr() != next(module_7.parameters()).data_ptr()
    for module in (module_2, module_7):
        state = module.get_state()
        assert state.keys() == source_state.keys()
        for key in state:
            torch.testing.assert_close(state[key], source_state[key])

    record = json.loads((output / "migration.json").read_text(encoding="utf-8"))
    assert record["migration"]["kind"] == "shared_to_independent_clone"
    assert record["migration"]["optimizer_state_migrated"] is False


def test_migrated_checkpoint_is_consumable_as_training_warm_start(tmp_path) -> None:
    source, _ = _legacy_algorithm_checkpoint(tmp_path)
    output = migrate_shared_checkpoint(
        source_checkpoint=source,
        output_path=tmp_path / "migration",
        target_agent_ids=[2, 7],
    )

    spec = build_rl_module_spec(
        [2, 7],
        initial_module_checkpoint=output / "multi_rl_module",
    )

    for agent_id in (2, 7):
        module_id = policy_id_for_agent(agent_id)
        assert spec.rl_module_specs[module_id].load_state_path == str(
            (output / "multi_rl_module" / module_id).resolve()
        )


def test_migration_refuses_overwrite(tmp_path) -> None:
    source, _ = _legacy_algorithm_checkpoint(tmp_path)
    output = tmp_path / "migration"
    output.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        migrate_shared_checkpoint(
            source_checkpoint=source,
            output_path=output,
            target_agent_ids=[2],
        )
