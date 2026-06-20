"""Explicit shared-policy to independent-module warm-start migration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ray.rllib.core.rl_module.rl_module import RLModule

from train.models.fcnet import RCSSPPOTorchRLModule
from train.train import build_rl_module_spec, policy_id_for_agent


LEGACY_MODULE_ID = "rcss_policy"


def _legacy_leaf(source: Path) -> Path:
    candidates = (
        source / "learner_group" / "learner" / "rl_module" / LEGACY_MODULE_ID,
        source / LEGACY_MODULE_ID,
        source,
    )
    for candidate in candidates:
        if (candidate / "metadata.json").is_file() and (
            candidate / "module_state.pkl"
        ).is_file():
            return candidate
    raise ValueError(f"Cannot locate legacy {LEGACY_MODULE_ID!r} module under {source}")


def migrate_shared_checkpoint(
    *,
    source_checkpoint: str | Path,
    output_path: str | Path,
    target_agent_ids: list[int] | tuple[int, ...],
) -> Path:
    source = Path(source_checkpoint).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite migration output: {output}")

    normalized_agents = tuple(sorted(target_agent_ids))
    target = build_rl_module_spec(normalized_agents).build()
    legacy = RLModule.from_checkpoint(_legacy_leaf(source))
    if not isinstance(legacy, RCSSPPOTorchRLModule):
        raise ValueError(
            f"Legacy module has unsupported class {type(legacy).__module__}.{type(legacy).__qualname__}"
        )

    source_state = legacy.get_state()
    for agent_id in normalized_agents:
        target[policy_id_for_agent(agent_id)].set_state(source_state)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    published = False
    try:
        module_output = temporary / "multi_rl_module"
        target.save_to_path(module_output)
        record = {
            "migration": {
                "kind": "shared_to_independent_clone",
                "source_checkpoint": source.as_posix(),
                "source_module": LEGACY_MODULE_ID,
                "target_modules": [
                    policy_id_for_agent(agent_id) for agent_id in normalized_agents
                ],
                "optimizer_state_migrated": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        }
        (temporary / "migration.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
        published = True
        return output
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def _agent_ids(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("agent ids must be comma-separated integers") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clone a legacy shared RLModule into independent per-unum modules"
    )
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--agent-ids", required=True, type=_agent_ids)
    args = parser.parse_args(argv)
    try:
        output = migrate_shared_checkpoint(
            source_checkpoint=args.source_checkpoint,
            output_path=args.output,
            target_agent_ids=args.agent_ids,
        )
    except (OSError, ValueError) as exc:
        print(f"Migration failed: {exc}")
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
