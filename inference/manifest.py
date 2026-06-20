"""Versioned inference-bundle manifest and ABI validation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rcss_env import obs as observation
from rcss_env.action import Action
from train.train import policy_id_for_agent


MANIFEST_FILE_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1
OBSERVATION_ABI_VERSION = "rcss-obs-v1"
ACTION_ABI_VERSION = "rcss-action-v1"
EXPECTED_MODULE_CLASS = "train.models.fcnet.RCSSPPOTorchRLModule"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceMetadata(StrictModel):
    checkpoint_uri: str = Field(min_length=1)
    experiment: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    training_iteration: int = Field(ge=0)
    metric: str = Field(min_length=1)
    metric_value: float
    metric_mode: Literal["min", "max"]
    git_commit: str = Field(min_length=1)


class ModelMetadata(StrictModel):
    algorithm: Literal["PPO"] = "PPO"
    module_class: str = EXPECTED_MODULE_CLASS
    module_ids: tuple[str, ...]
    stateful: Literal[False] = False


class PolicyTopology(StrictModel):
    mode: Literal["independent_per_unum"] = "independent_per_unum"
    agent_to_module: dict[str, str]

    @model_validator(mode="after")
    def validate_one_to_one_mapping(self) -> PolicyTopology:
        if not self.agent_to_module:
            raise ValueError("agent_to_module must not be empty")

        module_ids: list[str] = []
        for raw_agent_id, module_id in self.agent_to_module.items():
            try:
                unum = int(raw_agent_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Manifest agent id must be a positive integer string, got {raw_agent_id!r}"
                ) from exc
            if str(unum) != raw_agent_id or unum <= 0:
                raise ValueError(
                    f"Manifest agent id must use canonical positive integer form, got {raw_agent_id!r}"
                )
            expected_module_id = policy_id_for_agent(unum)
            if module_id != expected_module_id:
                raise ValueError(
                    f"Agent {unum} must map to {expected_module_id!r}, got {module_id!r}"
                )
            module_ids.append(module_id)

        if len(module_ids) != len(set(module_ids)):
            raise ValueError("Each v1 agent must map to a unique module")
        return self

    @property
    def agent_ids(self) -> tuple[int, ...]:
        return tuple(sorted(int(agent_id) for agent_id in self.agent_to_module))


class ObservationABI(StrictModel):
    abi_version: Literal["rcss-obs-v1"] = OBSERVATION_ABI_VERSION
    shape: tuple[int, ...] = (144,)
    dtype: Literal["float32"] = "float32"

    @model_validator(mode="after")
    def validate_current_observation_contract(self) -> ObservationABI:
        expected_shape = (observation.dim(),)
        if self.shape != expected_shape:
            raise ValueError(
                f"Observation shape must match current ABI {expected_shape}, got {self.shape}"
            )
        return self


class ActionABI(StrictModel):
    abi_version: Literal["rcss-action-v1"] = ACTION_ABI_VERSION
    names: tuple[str, ...] = Action.action_names()
    params_dim: int = Action.n_action_params()
    params_dtype: Literal["float32"] = "float32"
    params_low: float = Action.PARAM_LOW
    params_high: float = Action.PARAM_HIGH

    @model_validator(mode="after")
    def validate_current_action_contract(self) -> ActionABI:
        if self.names != Action.action_names():
            raise ValueError(
                f"Action names/order must be {Action.action_names()}, got {self.names}"
            )
        if self.params_dim != Action.n_action_params():
            raise ValueError(
                f"Action params_dim must be {Action.n_action_params()}, got {self.params_dim}"
            )
        if self.params_low != Action.PARAM_LOW or self.params_high != Action.PARAM_HIGH:
            raise ValueError(
                "Action parameter bounds do not match the current action ABI"
            )
        return self


class InferenceDefaults(StrictModel):
    default_mode: Literal["deterministic", "stochastic"] = "deterministic"
    continuous_postprocess: Literal["clip"] = "clip"


class BundleManifest(StrictModel):
    schema_version: Literal[1] = MANIFEST_SCHEMA_VERSION
    model_name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    model_version: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    created_at: datetime
    source: SourceMetadata
    model: ModelMetadata
    policy_topology: PolicyTopology
    observation: ObservationABI = Field(default_factory=ObservationABI)
    action: ActionABI = Field(default_factory=ActionABI)
    inference: InferenceDefaults = Field(default_factory=InferenceDefaults)

    @model_validator(mode="after")
    def validate_cross_section_contract(self) -> BundleManifest:
        if self.model.module_class != EXPECTED_MODULE_CLASS:
            raise ValueError(
                f"Unsupported module class {self.model.module_class!r}; "
                f"expected {EXPECTED_MODULE_CLASS!r}"
            )
        declared = self.model.module_ids
        if not declared:
            raise ValueError("model.module_ids must not be empty")
        if len(declared) != len(set(declared)):
            raise ValueError("model.module_ids must be unique")
        mapped = tuple(self.policy_topology.agent_to_module.values())
        if set(declared) != set(mapped):
            raise ValueError(
                "model.module_ids must exactly match policy_topology.agent_to_module values"
            )
        return self

    @property
    def agent_ids(self) -> tuple[int, ...]:
        return self.policy_topology.agent_ids

    def to_json(self) -> str:
        return self.model_dump_json(indent=2, exclude_none=True) + "\n"


def load_manifest(path: str | Path) -> BundleManifest:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / MANIFEST_FILE_NAME
    with manifest_path.open("r", encoding="utf-8") as fh:
        return BundleManifest.model_validate(json.load(fh))
