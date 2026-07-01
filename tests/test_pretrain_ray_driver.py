from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from pre_train.gen_datasets import GenDatasetCurriculumConfig, Image, SaveMode
from pre_train.gen_datasets.collector import DatasetRunResult
from pre_train.gen_datasets.ray_driver import (
    batch_summary_path,
    build_batch_summary,
    collect_match_once,
    collect_match_smoke,
    config_payload,
    run_distributed_collection,
    run_id_for_match,
    seed_for_match,
    write_batch_summary,
)


def _config(tmp_path: Path) -> GenDatasetCurriculumConfig:
    return GenDatasetCurriculumConfig(
        output_root=tmp_path,
        dataset_name="ray-test",
        save_mode=SaveMode.STATE,
        image_pool=[
            Image(image="HELIOS/helios-base"),
            Image(image="Cyrus2D/cyrus2024"),
        ],
        matches=2,
        random_seed=100,
        ray={"max_concurrent_matches": 2},
    )


def test_ray_driver_uses_deterministic_seed_and_run_id() -> None:
    assert seed_for_match(100, 3) == 103
    assert run_id_for_match("batch-a", 7) == "batch-a-match-000007"


def test_config_payload_round_trips_without_callable_mappings(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = config_payload(config)

    assert "left_team_name_mapping" not in payload
    assert "right_team_name_mapping" not in payload
    assert GenDatasetCurriculumConfig.model_validate(payload).image_pool[0].image == "HELIOS/helios-base"


def test_collect_match_once_returns_success_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FakeCollector:
        def __init__(self, config: GenDatasetCurriculumConfig, **kwargs: object) -> None:
            self.config = config
            self.kwargs = kwargs

        def collect_once(self, *, run_id: str) -> DatasetRunResult:
            run_dir = self.config.output_root / self.config.dataset_name / run_id
            run_dir.mkdir(parents=True)
            manifest_path = run_dir / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            return DatasetRunResult(
                run_dir=run_dir,
                manifest_path=manifest_path,
                cycles=[1, 2, 3],
                missing_cycles=[4],
                invalid_projection_cycles=[{"cycle": 5, "reason": "missing_agent_in_trainer_world_model"}],
            )

    monkeypatch.setattr("pre_train.gen_datasets.ray_driver.PretrainDatasetCollector", FakeCollector)

    with caplog.at_level(logging.INFO, logger="pre_train.gen_datasets.ray_driver"):
        result = collect_match_once(
            config_payload(_config(tmp_path)),
            match_index=1,
            seed=101,
            batch_id="batch-a",
        )

    assert result["status"] == "succeeded"
    assert result["run_id"] == "batch-a-match-000001"
    assert result["cycles"] == 3
    assert result["missing_cycles"] == [4]
    assert result["invalid_projection_cycles"] == [{"cycle": 5, "reason": "missing_agent_in_trainer_world_model"}]
    assert result["manifest_path"].endswith("/manifest.json")
    assert "Starting dataset match task batch_id=batch-a match_index=1" in caplog.text
    assert "Finished dataset match task batch_id=batch-a match_index=1" in caplog.text


def test_collect_match_once_returns_failure_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingCollector:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def collect_once(self, *, run_id: str) -> DatasetRunResult:
            raise RuntimeError(f"boom {run_id}")

    monkeypatch.setattr("pre_train.gen_datasets.ray_driver.PretrainDatasetCollector", FailingCollector)

    result = collect_match_once(
        config_payload(_config(tmp_path)),
        match_index=2,
        seed=102,
        batch_id="batch-a",
    )

    assert result["status"] == "failed"
    assert result["run_id"] == "batch-a-match-000002"
    assert result["error_type"] == "RuntimeError"
    assert "boom batch-a-match-000002" in result["error_message"]


def test_collect_match_smoke_writes_synthetic_manifest(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="pre_train.gen_datasets.ray_driver"):
        result = collect_match_smoke(
            config_payload(_config(tmp_path)),
            match_index=1,
            seed=101,
            batch_id="batch-a",
        )

    assert result["status"] == "succeeded"
    assert result["smoke_test"] is True
    manifest = json.loads(Path(str(result["manifest_path"])).read_text(encoding="utf-8"))
    assert manifest["kind"] == "rcss_pretrain_dataset_smoke_run"
    assert manifest["run_id"] == "batch-a-match-000001"
    assert "Finished synthetic dataset smoke task batch_id=batch-a match_index=1" in caplog.text


def test_write_batch_summary_records_successes_and_failures(tmp_path: Path) -> None:
    config = _config(tmp_path)
    results = [
        {"match_index": 2, "status": "failed"},
        {"match_index": 1, "status": "succeeded"},
    ]

    summary = build_batch_summary(config=config, batch_id="batch-a", results=results)
    assert summary["matches_requested"] == 2
    assert summary["matches_succeeded"] == 1
    assert summary["matches_failed"] == 1
    assert [item["match_index"] for item in summary["results"]] == [1, 2]

    path = write_batch_summary(config=config, batch_id="batch-a", results=results)
    assert path == batch_summary_path(config, "batch-a")
    assert json.loads(path.read_text(encoding="utf-8"))["matches_failed"] == 1


def test_run_distributed_collection_smoke_with_local_ray(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ray = pytest.importorskip("ray")
    config = _config(tmp_path)

    try:
        ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False, num_cpus=2)
    except PermissionError as exc:
        pytest.skip(f"Ray local mode cannot create sockets in this sandbox: {exc}")
    try:
        with caplog.at_level(logging.INFO, logger="pre_train.gen_datasets.ray_driver"):
            results, summary_path = run_distributed_collection(
                config,
                batch_id="batch-local",
                collect_fn=collect_match_smoke,
            )
    finally:
        ray.shutdown()

    assert [result["match_index"] for result in sorted(results, key=lambda item: item["match_index"])] == [1, 2]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["matches_requested"] == 2
    assert summary["matches_succeeded"] == 2
    assert summary["matches_failed"] == 0
    assert "Starting distributed dataset batch batch_id=batch-local" in caplog.text
    assert "Dataset batch progress batch_id=batch-local completed=2/2" in caplog.text
    assert "Finished distributed dataset batch batch_id=batch-local" in caplog.text
