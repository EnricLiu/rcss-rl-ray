"""Load and validate a local MultiRLModule inference checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from gymnasium import spaces
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule

from rcss_env import obs as observation
from rcss_env.action import Action
from rcss_env.action_mask import ActionMaskResolver
from train.models.fcnet import RCSSPPOTorchRLModule

from .manifest import BundleManifest, load_manifest


MODULE_DIR_NAME = "multi_rl_module"


class BundleValidationError(RuntimeError):
    """The model directory does not match the current experiment contract."""


class ModelLoadError(BundleValidationError):
    """The model checkpoint cannot be restored or executed."""


@dataclass(frozen=True)
class LoadedBundle:
    path: Path
    manifest: BundleManifest
    module: MultiRLModule
    device: torch.device


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise BundleValidationError("CUDA was requested but is not available")
    if requested not in {"cpu", "cuda"}:
        raise BundleValidationError(f"Unsupported inference device: {requested!r}")
    return torch.device(requested)


def validate_module_contract(
    multi_module: MultiRLModule,
    manifest: BundleManifest,
) -> None:
    expected_ids = set(manifest.model.module_ids)
    actual_ids = set(multi_module.keys())
    if actual_ids != expected_ids:
        raise BundleValidationError(
            f"Checkpoint module ids mismatch: expected={sorted(expected_ids)}, "
            f"actual={sorted(actual_ids)}"
        )

    for module_id in manifest.model.module_ids:
        module = multi_module[module_id]
        if not isinstance(module, RCSSPPOTorchRLModule):
            raise BundleValidationError(
                f"Module {module_id!r} has unsupported class "
                f"{type(module).__module__}.{type(module).__qualname__}"
            )
        if module.is_stateful() is not manifest.model.stateful:
            raise BundleValidationError(
                f"Module {module_id!r} stateful contract does not match manifest"
            )
        if not isinstance(module.observation_space, spaces.Box):
            raise BundleValidationError(
                f"Module {module_id!r} observation space must be Box"
            )
        if module.observation_space.shape != manifest.observation.shape:
            raise BundleValidationError(
                f"Module {module_id!r} observation shape mismatch: "
                f"{module.observation_space.shape}"
            )
        if not isinstance(module.action_space, spaces.Dict):
            raise BundleValidationError(
                f"Module {module_id!r} action space must be Dict"
            )
        if module.action_space != Action.space_schema():
            raise BundleValidationError(
                f"Module {module_id!r} action space does not match current ABI"
            )


def build_smoke_batch(
    module_ids: Iterable[str],
    device: torch.device,
) -> dict[str, dict[str, object]]:
    dash_mask = torch.nn.functional.one_hot(
        torch.tensor([Action.action_index(Action.DASH)], device=device),
        num_classes=Action.n_actions(),
    ).to(dtype=torch.float32)
    return {
        module_id: {
            Columns.OBS: {
                "obs": torch.zeros(
                    (1, observation.dim()), dtype=torch.float32, device=device
                ),
                ActionMaskResolver.OBSERVATION_KEY: dash_mask,
            }
        }
        for module_id in module_ids
    }


def smoke_inference(
    multi_module: MultiRLModule,
    manifest: BundleManifest,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        outputs = multi_module.forward_inference(
            build_smoke_batch(manifest.model.module_ids, device)
        )

    if set(outputs) != set(manifest.model.module_ids):
        raise BundleValidationError(
            "Smoke inference output module ids do not match the manifest"
        )

    logits_by_module: dict[str, torch.Tensor] = {}
    for module_id, output in outputs.items():
        logits = output.get(Columns.ACTION_DIST_INPUTS)
        if logits is None or logits.shape[0] != 1 or not torch.isfinite(logits).all():
            raise BundleValidationError(
                f"Module {module_id!r} returned invalid distribution inputs"
            )
        distribution_class = multi_module[module_id].get_inference_action_dist_cls()
        distribution = distribution_class.from_logits(logits)
        sample = distribution.to_deterministic().sample()
        if set(sample) != {"actions", "params"}:
            raise BundleValidationError(
                f"Module {module_id!r} returned invalid action structure"
            )
        action = sample["actions"]
        if tuple(action.shape) != (1,) or int(action.item()) != Action.action_index(
            Action.DASH
        ):
            raise BundleValidationError(
                f"Module {module_id!r} did not respect the smoke action mask"
            )
        params = sample["params"]
        valid_shape = tuple(params.shape) == (1, Action.n_action_params())
        if not valid_shape or not torch.isfinite(params).all():
            raise BundleValidationError(
                f"Module {module_id!r} returned invalid action parameters"
            )
        logits_by_module[module_id] = logits.detach().cpu()
    return logits_by_module


def validate_curriculum_agents(
    manifest: BundleManifest,
    agent_ids: Iterable[int],
) -> None:
    actual = tuple(sorted(agent_ids))
    if actual != manifest.agent_ids:
        raise BundleValidationError(
            f"Curriculum agent ids mismatch: expected={manifest.agent_ids}, actual={actual}"
        )


def load_bundle(
    bundle_path: str | Path,
    *,
    device: str = "auto",
) -> LoadedBundle:
    path = Path(bundle_path).expanduser().resolve()
    if not path.is_dir():
        raise BundleValidationError(f"Model directory does not exist: {path}")

    try:
        manifest = load_manifest(path)
        resolved_device = resolve_device(device)
    except BundleValidationError:
        raise
    except (OSError, ValueError) as exc:
        raise BundleValidationError("Could not read model metadata") from exc

    module_path = path / MODULE_DIR_NAME
    if not module_path.is_dir():
        raise BundleValidationError(f"Missing MultiRLModule checkpoint: {module_path}")
    try:
        multi_module = MultiRLModule.from_checkpoint(module_path)
    except Exception as exc:
        raise ModelLoadError("Failed to restore MultiRLModule checkpoint") from exc

    validate_module_contract(multi_module, manifest)
    try:
        for module_id in manifest.model.module_ids:
            multi_module[module_id].to(resolved_device)
            multi_module[module_id].eval()
        smoke_inference(multi_module, manifest, resolved_device)
    except BundleValidationError as exc:
        raise ModelLoadError("Loaded model failed smoke inference") from exc
    except Exception as exc:
        raise ModelLoadError("Failed to prepare loaded model") from exc

    return LoadedBundle(path, manifest, multi_module, resolved_device)
