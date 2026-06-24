from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from ipaddress import ip_address
from pathlib import Path
from uuid import uuid4

import numpy as np
from pydantic import IPvAnyAddress

from client.base.allocator import AllocatorClient
from client.room import RoomClient
from rcss_env.grpc_srv.proto import pb2
from rcss_env.grpc_srv.servicer import GameServicer, serve
from schema import TeamSide

from .config import GenDatasetCurriculumConfig, Image, SaveMode
from .projector import AgentIndexEntry, extract_agent_obs, team_agent_index
from .schema_builder import build_pretrain_schema
from .storage import append_world_model, write_agent_index, write_cycles, write_json, write_obs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetRunResult:
    run_dir: Path
    manifest_path: Path
    cycles: list[int]
    missing_cycles: list[int]


class PretrainDatasetCollector:
    def __init__(
        self,
        config: GenDatasetCurriculumConfig,
        *,
        allocator: AllocatorClient | None = None,
        servicer: GameServicer | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config
        self.allocator = allocator or AllocatorClient(config=config.allocator)
        self.servicer = servicer or GameServicer()
        self._rng = rng or random.Random(config.random_seed)

    def collect_once(
        self,
        *,
        left_image: Image | str | None = None,
        right_image: Image | str | None = None,
        run_id: str | None = None,
    ) -> DatasetRunResult:
        left, right = self._select_images(left_image=left_image, right_image=right_image)
        left_team_name = self._team_name(TeamSide.LEFT, left)
        right_team_name = self._team_name(TeamSide.RIGHT, right)
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        run_dir = self.config.output_root / self.config.dataset_name / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        server = None
        loop = None
        room = None
        try:
            server, actual_port, loop = serve(
                self.servicer,
                port=self.config.grpc_server.port,
                block=False,
            )
            grpc_host = self._advertised_host()
            schema = build_pretrain_schema(
                self.config,
                left_image=left,
                right_image=right,
                left_team_name=left_team_name,
                right_team_name=right_team_name,
                grpc_host=grpc_host,
                grpc_port=actual_port,
            )
            write_json(run_dir / "schema.json", schema.model_dump(mode="json", by_alias=True, exclude_none=True))

            room = self.allocator.request_room(schema)
            metrics_config = room.rcss.metrics_config()
            room.rcss.trainer.start()

            agent_index = self._agent_index(
                left,
                right,
                left_team_name=left_team_name,
                right_team_name=right_team_name,
            )
            write_agent_index(run_dir / "agent_index.json", agent_index)

            cycles: list[int] = []
            missing_cycles: list[int] = []
            obs_rows: list[np.ndarray] = []
            state_offsets: list[dict[str, int]] = []

            states_path = run_dir / "states.pb"
            should_save_state = self.config.save_mode in {SaveMode.STATE, SaveMode.BOTH}
            should_save_obs = self.config.save_mode in {SaveMode.OBS, SaveMode.BOTH}

            for cycle in range(1, self.config.time_up + 1):
                try:
                    wm = self.servicer.fetch_trainer_world_model(
                        cycle,
                        timeout=self.config.grpc_server.timeout,
                    )
                except Exception as exc:
                    logger.warning("Missing trainer world model cycle=%d: %s", cycle, exc)
                    missing_cycles.append(cycle)
                    continue

                cycles.append(int(wm.cycle))
                if should_save_state:
                    offset = append_world_model(states_path, wm)
                    state_offsets.append({"cycle": int(wm.cycle), "offset": offset})
                if should_save_obs:
                    obs_rows.append(self._obs_for_cycle(wm, agent_index))

                self.servicer.discard_trainer_before(cycle)

            if should_save_obs:
                write_obs(run_dir / "obs.npy", obs_rows)
            write_cycles(run_dir / "cycles.npy", cycles)
            if state_offsets:
                write_json(run_dir / "state_index.json", state_offsets)

            manifest = self._manifest(
                run_id=run_id,
                room=room,
                left_image=left,
                right_image=right,
                left_team_name=left_team_name,
                right_team_name=right_team_name,
                agent_index=agent_index,
                metrics_config=metrics_config.model_dump(mode="json", exclude_none=True),
                cycles=cycles,
                missing_cycles=missing_cycles,
            )
            manifest_path = run_dir / "manifest.json"
            write_json(manifest_path, manifest)
            return DatasetRunResult(
                run_dir=run_dir,
                manifest_path=manifest_path,
                cycles=cycles,
                missing_cycles=missing_cycles,
            )
        finally:
            if room is not None:
                room.release()
            if server is not None and loop is not None:
                self._stop_grpc_server(server, loop)

    def _advertised_host(self) -> IPvAnyAddress:
        host = self.config.grpc_server.host
        if str(host) not in {"0.0.0.0", "::"}:
            return host
        from ray.util import get_node_ip_address

        return ip_address(get_node_ip_address())

    def _select_images(
        self,
        *,
        left_image: Image | str | None,
        right_image: Image | str | None,
    ) -> tuple[Image, Image]:
        pool = list(self.config.image_pool)
        if not pool:
            raise ValueError("image_pool must contain at least one image")

        left = self._coerce_image(left_image) if left_image is not None else None
        right = self._coerce_image(right_image) if right_image is not None else None

        if left is None and right is None:
            if len(pool) == 1:
                return pool[0], pool[0]
            selected = self._rng.sample(pool, 2)
            return selected[0], selected[1]

        if left is None:
            left = self._pick_from_pool(excluding=right)
        if right is None:
            right = self._pick_from_pool(excluding=left)
        return left, right

    @staticmethod
    def _coerce_image(image: Image | str) -> Image:
        return image if isinstance(image, Image) else Image(image=image)

    def _pick_from_pool(self, *, excluding: Image | None) -> Image:
        candidates = [
            image for image in self.config.image_pool
            if excluding is None or image.image != excluding.image
        ]
        if not candidates:
            candidates = list(self.config.image_pool)
        if not candidates:
            raise ValueError("image_pool must contain at least one image")
        return self._rng.choice(candidates)

    def _team_name(self, side: TeamSide, image: Image) -> str:
        if side == TeamSide.LEFT:
            if self.config.left_team_name_mapping is not None:
                return self._fit_team_name(self.config.left_team_name_mapping(image), fallback="left-bots")
            return self._fit_team_name(self.config.left_team_name, fallback="left-bots")
        if self.config.right_team_name_mapping is not None:
            return self._fit_team_name(self.config.right_team_name_mapping(image), fallback="right-bots")
        return self._fit_team_name(self.config.right_team_name, fallback="right-bots")

    @staticmethod
    def _fit_team_name(name: str, *, fallback: str) -> str:
        if not name or not name.isascii():
            name = fallback
        return name[:16]

    def _agent_index(
        self,
        left: Image,
        right: Image,
        *,
        left_team_name: str,
        right_team_name: str,
    ) -> list[AgentIndexEntry]:
        return [
            *team_agent_index(
                side=TeamSide.LEFT,
                team_name=left_team_name,
                bot_image=left.image,
            ),
            *team_agent_index(
                side=TeamSide.RIGHT,
                team_name=right_team_name,
                bot_image=right.image,
            ),
        ]

    @staticmethod
    def _obs_for_cycle(wm: pb2.WorldModel, agent_index: list[AgentIndexEntry]) -> np.ndarray:
        return np.stack([extract_agent_obs(wm, entry) for entry in agent_index]).astype(np.float32, copy=False)

    def _manifest(
        self,
        *,
        run_id: str,
        room: RoomClient,
        left_image: Image,
        right_image: Image,
        left_team_name: str,
        right_team_name: str,
        agent_index: list[AgentIndexEntry],
        metrics_config: dict[str, object],
        cycles: list[int],
        missing_cycles: list[int],
    ) -> dict[str, object]:
        log_root = metrics_config.get("log_root")
        game_log_rel_dir = metrics_config.get("rcss_game_log_rel_dir")
        return {
            "schema_version": 1,
            "kind": "rcss_pretrain_dataset_run",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "save_mode": self.config.save_mode.value,
            "room": {
                "name": room.info.name,
                "host": getattr(room.info, "host", None),
                "base_url_rcss": room.info.base_url_rcss,
                "base_url_mc": room.info.base_url_mc,
            },
            "rcss_metrics_config": metrics_config,
            "action_log": {
                "log_root": log_root,
                "game_log_rel_dir": game_log_rel_dir,
                "expected_game_log_dir": (
                    f"{log_root.rstrip('/')}/{str(game_log_rel_dir).strip('/')}"
                    if isinstance(log_root, str) and isinstance(game_log_rel_dir, str)
                    else None
                ),
                "alignment_status": "pending_offline",
            },
            "teams": {
                "left": {"name": left_team_name, "bot_image": left_image.image},
                "right": {"name": right_team_name, "bot_image": right_image.image},
                "trainer_sides": [side.value for side in self.config.trainer.sides],
                "trainer_image": self.config.trainer.image,
            },
            "obs_abi": {
                "dtype": "float32",
                "shape": [144],
                "extractor": "rcss_env.obs.extract",
                "projection_version": 1,
            },
            "agent_index": [entry.to_json() for entry in agent_index],
            "cycles": {
                "start": min(cycles) if cycles else None,
                "end": max(cycles) if cycles else None,
                "count": len(cycles),
                "missing": missing_cycles,
            },
        }

    @staticmethod
    def _stop_grpc_server(server: object, loop: asyncio.AbstractEventLoop) -> None:
        async def _shutdown() -> None:
            await server.stop(grace=5)

        future = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
        future.result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
