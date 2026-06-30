from __future__ import annotations

import argparse
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .collector import PretrainDatasetCollector
from .config import GenDatasetCurriculumConfig
from .loader import load_gen_dataset_config
from .progress import manual_progress
from .storage import write_json

logger = logging.getLogger(__name__)


def _ensure_worker_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


def make_batch_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]


def seed_for_match(base_seed: int | None, match_index: int) -> int:
    if base_seed is None:
        return random.SystemRandom().randrange(0, 2**32)
    return int(base_seed) + int(match_index)


def run_id_for_match(batch_id: str, match_index: int) -> str:
    return f"{batch_id}-match-{match_index:06d}"


def batch_summary_path(config: GenDatasetCurriculumConfig, batch_id: str) -> Path:
    return config.output_root / config.dataset_name / batch_id / config.ray.summary_filename


def config_payload(config: GenDatasetCurriculumConfig) -> dict[str, Any]:
    return config.model_dump(mode="json", exclude={"left_team_name_mapping", "right_team_name_mapping"})


def collect_match_once(
    config_payload: dict[str, Any],
    *,
    match_index: int,
    seed: int,
    batch_id: str,
) -> dict[str, Any]:
    _ensure_worker_logging()
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = run_id_for_match(batch_id, match_index)
    logger.warning(
        "Starting dataset match task batch_id=%s match_index=%d run_id=%s seed=%d",
        batch_id,
        match_index,
        run_id,
        seed,
    )
    try:
        config = GenDatasetCurriculumConfig.model_validate(config_payload)
        result = PretrainDatasetCollector(
            config,
            rng=random.Random(seed),
        ).collect_once(run_id=run_id)
        finished_at = datetime.now(timezone.utc).isoformat()
        logger.warning(
            "Finished dataset match task batch_id=%s match_index=%d run_id=%s cycles=%d missing=%d",
            batch_id,
            match_index,
            run_id,
            len(result.cycles),
            len(result.missing_cycles),
        )
        return {
            "match_index": match_index,
            "run_id": run_id,
            "seed": seed,
            "status": "succeeded",
            "run_dir": result.run_dir.as_posix(),
            "manifest_path": result.manifest_path.as_posix(),
            "cycles": len(result.cycles),
            "missing_cycles": result.missing_cycles,
            "started_at": started_at,
            "finished_at": finished_at,
            "error_type": None,
            "error_message": None,
        }
    except Exception as exc:
        finished_at = datetime.now(timezone.utc).isoformat()
        logger.exception(
            "Dataset match task failed batch_id=%s match_index=%d run_id=%s",
            batch_id,
            match_index,
            run_id,
        )
        return {
            "match_index": match_index,
            "run_id": run_id,
            "seed": seed,
            "status": "failed",
            "run_dir": None,
            "manifest_path": None,
            "cycles": 0,
            "missing_cycles": [],
            "started_at": started_at,
            "finished_at": finished_at,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def collect_match_smoke(
    config_payload: dict[str, Any],
    *,
    match_index: int,
    seed: int,
    batch_id: str,
) -> dict[str, Any]:
    _ensure_worker_logging()
    started_at = datetime.now(timezone.utc).isoformat()
    config = GenDatasetCurriculumConfig.model_validate(config_payload)
    run_id = run_id_for_match(batch_id, match_index)
    run_dir = config.output_root / config.dataset_name / run_id
    manifest_path = run_dir / "manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "kind": "rcss_pretrain_dataset_smoke_run",
            "run_id": run_id,
            "match_index": match_index,
            "seed": seed,
            "created_at": started_at,
        },
    )
    finished_at = datetime.now(timezone.utc).isoformat()
    logger.warning(
        "Finished synthetic dataset smoke task batch_id=%s match_index=%d run_id=%s",
        batch_id,
        match_index,
        run_id,
    )
    return {
        "match_index": match_index,
        "run_id": run_id,
        "seed": seed,
        "status": "succeeded",
        "run_dir": run_dir.as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "cycles": 0,
        "missing_cycles": [],
        "started_at": started_at,
        "finished_at": finished_at,
        "error_type": None,
        "error_message": None,
        "smoke_test": True,
    }


def build_batch_summary(
    *,
    config: GenDatasetCurriculumConfig,
    batch_id: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    failures = [item for item in results if item.get("status") != "succeeded"]
    return {
        "schema_version": 1,
        "kind": "rcss_pretrain_dataset_batch",
        "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": config.dataset_name,
        "output_root": config.output_root.as_posix(),
        "matches_requested": config.matches,
        "matches_succeeded": len(results) - len(failures),
        "matches_failed": len(failures),
        "max_concurrent_matches": config.ray.max_concurrent_matches,
        "results": sorted(results, key=lambda item: int(item["match_index"])),
    }


def write_batch_summary(
    *,
    config: GenDatasetCurriculumConfig,
    batch_id: str,
    results: list[dict[str, Any]],
) -> Path:
    path = batch_summary_path(config, batch_id)
    write_json(path, build_batch_summary(config=config, batch_id=batch_id, results=results))
    return path


def _ray_task_options(config: GenDatasetCurriculumConfig) -> dict[str, Any]:
    options: dict[str, Any] = {
        "num_cpus": config.ray.num_cpus_per_match,
        "num_gpus": config.ray.num_gpus_per_match,
        "max_retries": config.ray.task_max_retries,
    }
    if config.ray.scheduling_strategy != "default":
        options["scheduling_strategy"] = config.ray.scheduling_strategy
    return options


def run_distributed_collection(
    config: GenDatasetCurriculumConfig,
    *,
    batch_id: str,
    collect_fn: Any = collect_match_once,
) -> tuple[list[dict[str, Any]], Path]:
    import ray

    started_at = time.monotonic()
    payload = config_payload(config)
    remote_collect = ray.remote(**_ray_task_options(config))(collect_fn)
    pending: list[tuple[int, int, Any]] = []
    results: list[dict[str, Any]] = []
    next_match_index = 1
    logger.warning(
        "Starting distributed dataset batch batch_id=%s matches=%d max_concurrent=%d "
        "num_cpus_per_match=%s num_gpus_per_match=%s",
        batch_id,
        config.matches,
        config.ray.max_concurrent_matches,
        config.ray.num_cpus_per_match,
        config.ray.num_gpus_per_match,
    )

    def submit_next() -> None:
        nonlocal next_match_index
        seed = seed_for_match(config.random_seed, next_match_index)
        ref = remote_collect.remote(
            payload,
            match_index=next_match_index,
            seed=seed,
            batch_id=batch_id,
        )
        pending.append((next_match_index, seed, ref))
        logger.warning(
            "Submitted dataset match task batch_id=%s match_index=%d seed=%d pending=%d completed=%d/%d",
            batch_id,
            next_match_index,
            seed,
            len(pending),
            len(results),
            config.matches,
        )
        next_match_index += 1

    while next_match_index <= config.matches and len(pending) < config.ray.max_concurrent_matches:
        submit_next()

    with manual_progress(
        progress=config.progress,
        total=config.matches,
        desc=f"batch {batch_id}",
        unit="match",
    ) as progress_bar:
        while pending:
            ready_refs, _ = ray.wait([ref for _, _, ref in pending], num_returns=1)
            ready_ref = ready_refs[0]
            match_index, seed, _ = next(item for item in pending if item[2] == ready_ref)
            pending = [item for item in pending if item[2] != ready_ref]
            try:
                result = ray.get(ready_ref)
            except Exception as exc:
                logger.exception(
                    "Dataset match task crashed before returning structured result batch_id=%s match_index=%d",
                    batch_id,
                    match_index,
                )
                result = {
                    "match_index": match_index,
                    "run_id": run_id_for_match(batch_id, match_index),
                    "seed": seed,
                    "status": "failed",
                    "run_dir": None,
                    "manifest_path": None,
                    "cycles": 0,
                    "missing_cycles": [],
                    "started_at": None,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            results.append(result)
            if progress_bar is not None:
                progress_bar.update(1)
            _log_batch_progress(
                batch_id=batch_id,
                config=config,
                results=results,
                pending_count=len(pending),
                started_at=started_at,
            )

            if next_match_index <= config.matches:
                submit_next()

    summary_path = write_batch_summary(config=config, batch_id=batch_id, results=results)
    failures = [item for item in results if item.get("status") != "succeeded"]
    logger.warning(
        "Finished distributed dataset batch batch_id=%s summary=%s succeeded=%d failed=%d elapsed_s=%.1f",
        batch_id,
        summary_path,
        len(results) - len(failures),
        len(failures),
        time.monotonic() - started_at,
    )
    return results, summary_path


def _log_batch_progress(
    *,
    batch_id: str,
    config: GenDatasetCurriculumConfig,
    results: list[dict[str, Any]],
    pending_count: int,
    started_at: float,
) -> None:
    completed = len(results)
    if not config.progress.enabled:
        return
    if (
        completed == 1
        or completed == config.matches
        or completed % config.progress.match_log_interval == 0
    ):
        failures = sum(1 for item in results if item.get("status") != "succeeded")
        elapsed_s = max(time.monotonic() - started_at, 0.001)
        logger.warning(
            "Dataset batch progress batch_id=%s completed=%d/%d failed=%d pending=%d "
            "matches_per_s=%.2f",
            batch_id,
            completed,
            config.matches,
            failures,
            pending_count,
            completed / elapsed_s,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RCSS pretraining datasets with Ray tasks.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML, JSON, or TOML GenDatasetCurriculumConfig.",
    )
    parser.add_argument("--batch-id", type=str, default=None, help="Optional deterministic batch id.")
    parser.add_argument("--ray-address", type=str, default="auto", help="Ray address passed to ray.init().")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run synthetic Ray tasks without allocating RCSS rooms.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_gen_dataset_config(args.config)
    batch_id = args.batch_id or make_batch_id()

    import ray

    logger.warning("Connecting to Ray address=%s batch_id=%s smoke_test=%s", args.ray_address, batch_id, args.smoke_test)
    ray.init(address=args.ray_address)
    try:
        collect_fn = collect_match_smoke if args.smoke_test else collect_match_once
        results, summary_path = run_distributed_collection(
            config,
            batch_id=batch_id,
            collect_fn=collect_fn,
        )
    finally:
        ray.shutdown()

    print(summary_path)
    return 1 if any(item.get("status") != "succeeded" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
