"""Extract a local MultiRLModule checkpoint for experiment inference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule

from train.config import TrainConfig, load_train_config
from train.factory import build_env_config
from train.train import policy_id_for_agent

from .loader import (
    MODULE_DIR_NAME,
    BundleValidationError,
    load_bundle,
    smoke_inference,
    validate_module_contract,
)
from .manifest import (
    ActionABI,
    BundleManifest,
    InferenceDefaults,
    ModelMetadata,
    ObservationABI,
    PolicyTopology,
    SourceMetadata,
)


class BestCheckpointMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    checkpoint_uri: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    metric_value: float
    metric_mode: Literal["min", "max"] = "max"
    training_iteration: int = Field(ge=0)
    created_at: datetime


def load_best_checkpoint_metadata(path: str | Path) -> BestCheckpointMetadata:
    with Path(path).open("r", encoding="utf-8") as fh:
        return BestCheckpointMetadata.model_validate(json.load(fh))


def current_git_commit(workdir: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BundleValidationError("Cannot resolve git commit") from exc
    return result.stdout.strip()


def experiment_config(train_config: TrainConfig) -> dict[str, Any]:
    """Keep the model and curriculum settings needed to reproduce an experiment."""
    return {
        "algorithm": train_config.algo,
        "ppo": train_config.ppo.model_dump(mode="json"),
        "curriculum": train_config.curriculum_config.model_dump(
            mode="json", exclude={"grpc_server"}
        ),
    }


def _assert_same_outputs(
    source: dict[str, torch.Tensor],
    exported: dict[str, torch.Tensor],
) -> None:
    if set(source) != set(exported):
        raise BundleValidationError("Source and exported module sets differ")
    for module_id in source:
        try:
            torch.testing.assert_close(source[module_id], exported[module_id])
        except AssertionError as exc:
            raise BundleValidationError(
                f"Exported module {module_id!r} differs from the source checkpoint"
            ) from exc


def export_bundle(
    *,
    checkpoint_path: str | Path,
    train_config_path: str | Path,
    output_root: str | Path,
    model_name: str,
    model_version: str,
    experiment: str,
    trial_id: str,
    training_iteration: int,
    metric_name: str,
    metric_value: float,
    metric_mode: str = "max",
    git_commit: str | None = None,
) -> Path:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise BundleValidationError(f"Checkpoint does not exist: {checkpoint}")
    module_source = checkpoint / "learner_group" / "learner" / "rl_module"
    if not module_source.is_dir():
        raise BundleValidationError(
            f"Algorithm checkpoint does not contain {module_source}"
        )

    train_config = load_train_config(train_config_path)
    agent_ids = tuple(sorted(build_env_config(train_config).curriculum.agent_unums()))
    mapping = {str(agent_id): policy_id_for_agent(agent_id) for agent_id in agent_ids}
    manifest = BundleManifest(
        model_name=model_name,
        model_version=model_version,
        created_at=datetime.now(timezone.utc),
        source=SourceMetadata(
            checkpoint_uri=checkpoint.as_posix(),
            experiment=experiment,
            trial_id=trial_id,
            training_iteration=training_iteration,
            metric=metric_name,
            metric_value=metric_value,
            metric_mode=metric_mode,
            git_commit=git_commit
            or current_git_commit(Path(__file__).resolve().parents[1]),
        ),
        model=ModelMetadata(module_ids=tuple(mapping.values())),
        policy_topology=PolicyTopology(agent_to_module=mapping),
        observation=ObservationABI(),
        action=ActionABI(),
        inference=InferenceDefaults(),
    )

    source_module = MultiRLModule.from_checkpoint(module_source)
    validate_module_contract(source_module, manifest)
    for module_id in manifest.model.module_ids:
        source_module[module_id].to(torch.device("cpu"))
        source_module[module_id].eval()
    source_outputs = smoke_inference(source_module, manifest, torch.device("cpu"))

    output = Path(output_root).expanduser() / model_name / model_version
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite model directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{model_version}.", dir=output.parent))
    try:
        shutil.copytree(module_source, temporary / MODULE_DIR_NAME)
        (temporary / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
        (temporary / "experiment.yaml").write_text(
            yaml.safe_dump(experiment_config(train_config), sort_keys=False),
            encoding="utf-8",
        )
        loaded = load_bundle(temporary, device="cpu")
        _assert_same_outputs(
            source_outputs,
            smoke_inference(loaded.module, loaded.manifest, loaded.device),
        )
        os.replace(temporary, output)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a local RLlib checkpoint for inference"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--best-checkpoint-json", type=Path)
    parser.add_argument("--train-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--experiment")
    parser.add_argument("--trial-id")
    parser.add_argument("--training-iteration", type=int)
    parser.add_argument("--metric-name")
    parser.add_argument("--metric-value", type=float)
    parser.add_argument("--metric-mode", choices=["min", "max"])
    parser.add_argument("--git-commit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.best_checkpoint_json:
            best = load_best_checkpoint_metadata(args.best_checkpoint_json)
            checkpoint = best.checkpoint_uri
            experiment = best.experiment
            trial_id = best.trial_id
            training_iteration = best.training_iteration
            metric_name = best.metric
            metric_value = best.metric_value
            metric_mode = best.metric_mode
        else:
            required = (
                args.experiment,
                args.trial_id,
                args.training_iteration,
                args.metric_value,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "Manual export requires --experiment, --trial-id, "
                    "--training-iteration and --metric-value"
                )
            checkpoint = args.checkpoint
            experiment = args.experiment
            trial_id = args.trial_id
            training_iteration = args.training_iteration
            metric_name = args.metric_name or "checkpoint_score"
            metric_value = args.metric_value
            metric_mode = args.metric_mode or "max"
        output = export_bundle(
            checkpoint_path=checkpoint,
            train_config_path=args.train_config,
            output_root=args.output,
            model_name=args.model_name,
            model_version=args.version,
            experiment=experiment,
            trial_id=trial_id,
            training_iteration=training_iteration,
            metric_name=metric_name,
            metric_value=metric_value,
            metric_mode=metric_mode,
            git_commit=args.git_commit,
        )
    except (BundleValidationError, FileExistsError, ValueError, OSError) as exc:
        print(f"Export failed: {exc}")
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
