from __future__ import annotations

import asyncio
import json
import logging
import random
import sys
from ipaddress import IPv4Address
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from httpx import Client, MockTransport, Request, Response

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.base.rcss import RcssClient
from pre_train.gen_datasets import GenDatasetCurriculumConfig, Image, PretrainDatasetCollector, SaveMode
from pre_train.gen_datasets.loader import load_config_mapping, load_gen_dataset_config
from pre_train.gen_datasets.projector import AgentIndexEntry, extract_agent_obs, project_global_world_model
from pre_train.gen_datasets.schema_builder import build_pretrain_schema
from pre_train.gen_datasets.storage import append_world_model, iter_world_models
from rcss_env.grpc_srv.proto import pb2
from rcss_env.grpc_srv.servicer import GameServicer
from schema import SspAgentPolicy, TeamSide, TrainerSchema


def envelope(payload: object, *, success: bool = True) -> dict[str, object]:
    return {
        "id": 1,
        "success": success,
        "payload": payload,
        "created_at": "2026-06-24T00:00:00Z",
    }


def test_trainer_schema_serializes_like_allocator_trainer_v1() -> None:
    trainer = TrainerSchema(
        policy=SspAgentPolicy(
            image="CLSFramework/soccer-simulation-proxy",
            grpc_host=IPv4Address("127.0.0.1"),
            grpc_port=43123,
        )
    )

    assert trainer.model_dump(mode="json") == {
        "policy": {
            "kind": "agent",
            "image": "CLSFramework/soccer-simulation-proxy",
            "agent": "ssp",
            "grpc_host": "127.0.0.1",
            "grpc_port": 43123,
        }
    }


def test_build_pretrain_schema_uses_bot_players_and_ssp_trainer() -> None:
    config = GenDatasetCurriculumConfig(
        save_mode=SaveMode.STATE,
        image_pool=[Image(image="HELIOS/helios-base")],
        time_up=123,
    )

    schema = build_pretrain_schema(
        config,
        left_image=Image(image="HELIOS/helios-base"),
        right_image=Image(image="HELIOS/helios-base"),
        grpc_host=IPv4Address("127.0.0.1"),
        grpc_port=43123,
    )
    payload = schema.model_dump(mode="json", by_alias=True, exclude_none=True)

    assert payload["teams"]["left"]["trainer"]["policy"]["agent"] == "ssp"
    assert payload["teams"]["left"]["trainer"]["policy"]["grpc_port"] == 43123
    assert "trainer" not in payload["teams"]["right"]
    assert {player["policy"]["kind"] for player in payload["teams"]["left"]["players"]} == {"bot"}
    assert payload["stopping"]["time_up"] == 123

    with pytest.raises(ValueError, match="exactly one agentic team"):
        _ = schema.teams.agent_team


def test_collector_randomly_selects_images_from_pool() -> None:
    config = GenDatasetCurriculumConfig(
        image_pool=[
            Image(image="A/a"),
            Image(image="B/b"),
            Image(image="C/c"),
        ],
    )
    collector = PretrainDatasetCollector(config, allocator=object(), servicer=object(), rng=random.Random(0))

    left, right = collector._select_images(left_image=None, right_image=None)

    assert (left.image, right.image) == ("B/b", "C/c")
    assert left.image != right.image


def test_collector_uses_random_image_when_only_one_side_is_overridden() -> None:
    config = GenDatasetCurriculumConfig(
        image_pool=[
            Image(image="A/a"),
            Image(image="B/b"),
            Image(image="C/c"),
        ],
    )
    collector = PretrainDatasetCollector(config, allocator=object(), servicer=object(), rng=random.Random(0))

    left, right = collector._select_images(left_image=Image(image="B/b"), right_image=None)

    assert left.image == "B/b"
    assert right.image in {"A/a", "C/c"}


def test_load_gen_dataset_config_supports_yaml_json_and_toml(tmp_path: Path) -> None:
    payload = {
        "type": "gen_dataset",
        "dataset_name": "loader-test",
        "image_pool": ["HELIOS/helios-base", "Cyrus2D/cyrus2024"],
        "time_up": 12,
        "matches": 2,
        "progress": {"cycle_log_interval": 3, "match_log_interval": 2, "tqdm": "never"},
        "ray": {"max_concurrent_matches": 2},
    }

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "type: gen_dataset",
                "dataset_name: loader-test",
                "image_pool:",
                "  - HELIOS/helios-base",
                "  - Cyrus2D/cyrus2024",
                "time_up: 12",
                "matches: 2",
                "progress:",
                "  cycle_log_interval: 3",
                "  match_log_interval: 2",
                "  tqdm: never",
                "ray:",
                "  max_concurrent_matches: 2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    json_path = tmp_path / "config.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    toml_path = tmp_path / "config.toml"
    toml_path.write_text(
        "\n".join(
            [
                'type = "gen_dataset"',
                'dataset_name = "loader-test"',
                'image_pool = ["HELIOS/helios-base", "Cyrus2D/cyrus2024"]',
                "time_up = 12",
                "matches = 2",
                "[progress]",
                "cycle_log_interval = 3",
                "match_log_interval = 2",
                'tqdm = "never"',
                "[ray]",
                "max_concurrent_matches = 2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert load_config_mapping(yaml_path)["dataset_name"] == "loader-test"
    assert load_gen_dataset_config(yaml_path).ray.max_concurrent_matches == 2
    assert load_gen_dataset_config(yaml_path).progress.cycle_log_interval == 3
    assert load_gen_dataset_config(json_path).image_pool[1].image == "Cyrus2D/cyrus2024"
    assert load_gen_dataset_config(toml_path).time_up == 12


def test_ray_concurrency_must_not_exceed_matches() -> None:
    with pytest.raises(ValueError, match="max_concurrent_matches"):
        GenDatasetCurriculumConfig(
            matches=1,
            ray={"max_concurrent_matches": 2},
        )


def test_rcss_metrics_config_parses_log_root() -> None:
    def handler(request: Request) -> Response:
        assert request.method == "GET"
        assert request.url.path == "/metrics/config"
        return Response(
            200,
            json=envelope(
                {
                    "log_root": "/var/log/rcss",
                    "rcss_game_log_rel_dir": "games",
                    "rcss_stdio_log_rel_path": "stdio/rcss.log",
                    "half_time_auto_start": True,
                    "always_log_stdout": False,
                }
            ),
        )

    rcss = RcssClient(
        "http://rcss.test",
        client=Client(base_url="http://rcss.test", transport=MockTransport(handler)),
    )

    config = rcss.metrics_config()

    assert config.log_root == "/var/log/rcss"
    assert config.rcss_game_log_rel_dir == "games"
    assert config.half_time_auto_start is True


def test_rcss_metrics_config_accepts_numeric_half_time_auto_start() -> None:
    def handler(request: Request) -> Response:
        assert request.method == "GET"
        assert request.url.path == "/metrics/config"
        return Response(
            200,
            json=envelope(
                {
                    "log_root": "/var/log/rcss",
                    "rcss_game_log_rel_dir": "games",
                    "half_time_auto_start": 3000,
                }
            ),
        )

    rcss = RcssClient(
        "http://rcss.test",
        client=Client(base_url="http://rcss.test", transport=MockTransport(handler)),
    )

    config = rcss.metrics_config()

    assert config.log_root == "/var/log/rcss"
    assert config.half_time_auto_start == 3000


def test_servicer_stores_trainer_world_model_exact_cycle() -> None:
    async def exercise() -> None:
        servicer = GameServicer()
        state = pb2.State(world_model=pb2.WorldModel(cycle=7, game_mode_type=pb2.PlayOn))

        await servicer.GetTrainerActions(state, None)
        fetched = await servicer._GameServicer__fetch_trainer_world_model(7, timeout=0.1)

        assert fetched.cycle == 7
        assert servicer.debug_snapshot()["trainer_buffer"]["buffered_cycles"] == [7]
        await servicer._trainer_buffer.reset()

    asyncio.run(exercise())


def _player(unum: int, side: int, x: float, y: float) -> pb2.Player:
    player = pb2.Player()
    player.uniform_number = unum
    player.side = side
    player.position.x = x
    player.position.y = y
    player.velocity.x = 0.1 * unum
    player.velocity.y = -0.1 * unum
    return player


def _global_world_model() -> pb2.WorldModel:
    wm = pb2.WorldModel(cycle=11, our_side=pb2.LEFT, game_mode_type=pb2.PlayOn)
    wm.our_team_name = "left-bots"
    wm.their_team_name = "right-bots"
    wm.ball.position.x = 1.0
    wm.ball.position.y = 2.0
    wm.ball.velocity.x = 0.2
    wm.ball.velocity.y = -0.3
    for unum in range(1, 12):
        wm.our_players_dict[unum].CopyFrom(_player(unum, pb2.LEFT, float(unum), 0.0))
        wm.their_players_dict[unum].CopyFrom(_player(unum, pb2.RIGHT, -float(unum), 1.0))
    return wm


def test_project_global_world_model_and_extract_obs() -> None:
    wm = _global_world_model()

    projected = project_global_world_model(wm, side=TeamSide.RIGHT, unum=2)
    assert projected.self.uniform_number == 2
    assert projected.self.side == pb2.RIGHT
    assert projected.our_team_name == "right-bots"
    assert 2 in projected.our_players_dict

    obs = extract_agent_obs(
        wm,
        AgentIndexEntry(
            side=TeamSide.RIGHT,
            unum=2,
            team_name="right-bots",
            bot_image="HELIOS/helios-base",
        ),
    )
    assert obs.shape == (144,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


def test_world_model_stream_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "states.pb"
    first = pb2.WorldModel(cycle=1)
    second = pb2.WorldModel(cycle=2)

    assert append_world_model(path, first) == 0
    assert append_world_model(path, second) > 0

    assert [wm.cycle for wm in iter_world_models(path)] == [1, 2]


def test_collector_writes_manifest_with_log_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FakeMetricsConfig:
        def model_dump(self, **_: object) -> dict[str, object]:
            return {
                "log_root": "/var/log/rcss",
                "rcss_game_log_rel_dir": "games",
                "rcss_stdio_log_rel_path": "stdio/rcss.log",
            }

    class FakeRoom:
        def __init__(self) -> None:
            self.info = SimpleNamespace(
                name="room-a",
                host="127.0.0.1",
                base_url_rcss="http://127.0.0.1:6666",
                base_url_mc="http://127.0.0.1:7777",
            )
            self.released = False
            self.started = False
            self.rcss = SimpleNamespace(
                metrics_config=lambda: FakeMetricsConfig(),
                trainer=SimpleNamespace(start=self._start),
            )

        def _start(self) -> None:
            self.started = True

        def release(self) -> None:
            self.released = True

    class FakeAllocator:
        def __init__(self) -> None:
            self.requested_schema = None
            self.room = FakeRoom()

        def request_room(self, schema: object) -> FakeRoom:
            self.requested_schema = schema
            return self.room

    class FakeServicer:
        def __init__(self) -> None:
            self.discarded: list[int] = []

        def fetch_trainer_world_model(self, cycle: int, timeout: float) -> pb2.WorldModel:
            return pb2.WorldModel(cycle=cycle, our_side=pb2.LEFT, game_mode_type=pb2.PlayOn)

        def discard_trainer_before(self, cycle: int) -> None:
            self.discarded.append(cycle)

    monkeypatch.setattr("pre_train.gen_datasets.collector.serve", lambda *_args, **_kwargs: (object(), 43123, object()))
    monkeypatch.setattr(PretrainDatasetCollector, "_advertised_host", lambda self: IPv4Address("127.0.0.1"))
    monkeypatch.setattr(PretrainDatasetCollector, "_stop_grpc_server", staticmethod(lambda *_args: None))

    config = GenDatasetCurriculumConfig(
        output_root=tmp_path,
        dataset_name="pretrain-test",
        save_mode=SaveMode.STATE,
        image_pool=[
            Image(image="HELIOS/helios-base"),
            Image(image="RoboCIn/RoboCIn2025"),
            Image(image="YuShan/YuShan2025"),
        ],
        random_seed=0,
        time_up=2,
        progress={"cycle_log_interval": 1, "tqdm": "never"},
    )
    allocator = FakeAllocator()
    servicer = FakeServicer()

    with caplog.at_level(logging.INFO, logger="pre_train.gen_datasets.collector"):
        result = PretrainDatasetCollector(
            config,
            allocator=allocator,
            servicer=servicer,
        ).collect_once(run_id="run-a")

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    schema_payload = json.loads((result.run_dir / "schema.json").read_text(encoding="utf-8"))
    agent_index = json.loads((result.run_dir / "agent_index.json").read_text(encoding="utf-8"))
    assert manifest["action_log"]["log_root"] == "/var/log/rcss"
    assert manifest["action_log"]["expected_game_log_dir"] == "/var/log/rcss/games"
    assert manifest["action_log"]["alignment_status"] == "pending_offline"
    assert manifest["teams"]["left"]["bot_image"] == "RoboCIn/RoboCIn2025"
    assert manifest["teams"]["right"]["bot_image"] == "YuShan/YuShan2025"
    assert schema_payload["teams"]["left"]["players"][0]["policy"]["image"] == "RoboCIn/RoboCIn2025"
    assert schema_payload["teams"]["right"]["players"][0]["policy"]["image"] == "YuShan/YuShan2025"
    assert agent_index[0]["bot_image"] == "RoboCIn/RoboCIn2025"
    assert agent_index[-1]["bot_image"] == "YuShan/YuShan2025"
    assert manifest["cycles"]["count"] == 2
    assert allocator.room.started is True
    assert allocator.room.released is True
    assert servicer.discarded == [1, 2]
    assert [wm.cycle for wm in iter_world_models(result.run_dir / "states.pb")] == [1, 2]
    assert allocator.requested_schema.teams.left.trainer.policy.grpc_port == 43123
    assert "Starting pretrain dataset run run_id=run-a" in caplog.text
    assert "Dataset run progress run_id=run-a cycle=1/2" in caplog.text
    assert "Dataset run progress run_id=run-a cycle=2/2" in caplog.text
    assert "Finished pretrain dataset run run_id=run-a" in caplog.text


def test_collector_skips_unprojectable_obs_cycles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMetricsConfig:
        def model_dump(self, **_: object) -> dict[str, object]:
            return {
                "log_root": "/var/log/rcss",
                "rcss_game_log_rel_dir": "games",
            }

    class FakeRoom:
        def __init__(self) -> None:
            self.info = SimpleNamespace(
                name="room-a",
                host="127.0.0.1",
                base_url_rcss="http://127.0.0.1:6666",
                base_url_mc="http://127.0.0.1:7777",
            )
            self.rcss = SimpleNamespace(
                metrics_config=lambda: FakeMetricsConfig(),
                trainer=SimpleNamespace(start=lambda: None),
            )

        def release(self) -> None:
            pass

    class FakeAllocator:
        def request_room(self, schema: object) -> FakeRoom:
            return FakeRoom()

    class FakeServicer:
        def __init__(self) -> None:
            self.discarded: list[int] = []

        def fetch_trainer_world_model(self, cycle: int, timeout: float) -> pb2.WorldModel:
            if cycle == 1:
                return pb2.WorldModel(cycle=cycle, our_side=pb2.LEFT, game_mode_type=pb2.PlayOn)
            wm = _global_world_model()
            wm.cycle = cycle
            return wm

        def discard_trainer_before(self, cycle: int) -> None:
            self.discarded.append(cycle)

    monkeypatch.setattr("pre_train.gen_datasets.collector.serve", lambda *_args, **_kwargs: (object(), 43123, object()))
    monkeypatch.setattr(PretrainDatasetCollector, "_advertised_host", lambda self: IPv4Address("127.0.0.1"))
    monkeypatch.setattr(PretrainDatasetCollector, "_stop_grpc_server", staticmethod(lambda *_args: None))

    config = GenDatasetCurriculumConfig(
        output_root=tmp_path,
        dataset_name="pretrain-test",
        save_mode=SaveMode.OBS,
        image_pool=[Image(image="HELIOS/helios-base")],
        time_up=2,
        progress={"cycle_log_interval": 1, "tqdm": "never"},
    )
    servicer = FakeServicer()

    result = PretrainDatasetCollector(
        config,
        allocator=FakeAllocator(),
        servicer=servicer,
    ).collect_once(run_id="run-a")

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    obs = np.load(result.run_dir / "obs.npy")
    cycles = np.load(result.run_dir / "cycles.npy")

    assert result.cycles == [2]
    assert cycles.tolist() == [2]
    assert obs.shape == (1, 22, 144)
    assert manifest["cycles"]["count"] == 1
    assert manifest["cycles"]["invalid_projection"][0]["cycle"] == 1
    assert manifest["cycles"]["invalid_projection"][0]["reason"] == "missing_agent_in_trainer_world_model"
    assert len(manifest["cycles"]["invalid_projection"][0]["missing_agents"]) == 22
    assert result.invalid_projection_cycles == manifest["cycles"]["invalid_projection"]
    assert servicer.discarded == [1, 2]
