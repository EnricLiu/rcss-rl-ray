from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
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
from .progress import iter_progress
from .projector import AgentIndexEntry, extract_agent_obs, missing_projectable_agents, team_agent_index
from .schema_builder import build_pretrain_schema
from .storage import append_world_model, write_agent_index, write_cycles, write_json, write_obs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetRunResult:
    run_dir: Path
    manifest_path: Path
    cycles: list[int]
    missing_cycles: list[int]
    invalid_projection_cycles: list[dict[str, object]] = field(default_factory=list)


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
        started_at = time.monotonic()
        logger.warning(
            "Starting pretrain dataset run run_id=%s output=%s save_mode=%s time_up=%d "
            "left_image=%s right_image=%s left_team=%s right_team=%s",
            run_id,
            run_dir,
            self.config.save_mode.value,
            self.config.time_up,
            left.image,
            right.image,
            left_team_name,
            right_team_name,
        )

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
            logger.warning(
                "Started dataset gRPC server run_id=%s advertised_host=%s port=%d",
                run_id,
                grpc_host,
                actual_port,
            )
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
            logger.warning("Wrote dataset schema run_id=%s path=%s", run_id, run_dir / "schema.json")

            room = self.allocator.request_room(schema)
            logger.warning(
                "Allocated dataset room run_id=%s room=%s rcss=%s mc=%s",
                run_id,
                room.info.name,
                room.info.base_url_rcss,
                room.info.base_url_mc,
            )
            metrics_config = room.rcss.metrics_config()
            logger.warning(
                "Fetched RCSS metrics config run_id=%s log_root=%s game_log_rel_dir=%s",
                run_id,
                getattr(metrics_config, "log_root", None),
                getattr(metrics_config, "rcss_game_log_rel_dir", None),
            )
            room.rcss.trainer.start()
            logger.warning("Started SSP trainer run_id=%s room=%s", run_id, room.info.name)

            agent_index = self._agent_index(
                left,
                right,
                left_team_name=left_team_name,
                right_team_name=right_team_name,
            )
            write_agent_index(run_dir / "agent_index.json", agent_index)
            logger.warning("Wrote agent index run_id=%s agents=%d", run_id, len(agent_index))

            cycles: list[int] = []
            missing_cycles: list[int] = []
            invalid_projection_cycles: list[dict[str, object]] = []
            obs_rows: list[np.ndarray] = []
            state_offsets: list[dict[str, int]] = []

            states_path = run_dir / "states.pb"
            should_save_state = self.config.save_mode in {SaveMode.STATE, SaveMode.BOTH}
            should_save_obs = self.config.save_mode in {SaveMode.OBS, SaveMode.BOTH}

            cycle_iter = iter_progress(
                range(1, self.config.time_up + 1),
                progress=self.config.progress,
                total=self.config.time_up,
                desc=f"dataset {run_id}",
                unit="cycle",
            )
            for cycle in cycle_iter:
                try:
                    wm = self.servicer.fetch_trainer_world_model(
                        cycle,
                        timeout=self.config.grpc_server.timeout,
                    )
                except Exception as exc:
                    logger.warning("Missing trainer world model cycle=%d: %s", cycle, exc)
                    missing_cycles.append(cycle)
                    continue

                obs_row: np.ndarray | None = None
                if should_save_obs:
                    invalid_projection = self._invalid_projection_cycle(wm, agent_index)
                    if invalid_projection is not None:
                        invalid_projection_cycles.append(invalid_projection)
                        logger.warning(
                            "Skipping unprojectable trainer world model run_id=%s cycle=%d "
                            "missing_agents=%d sample=%s",
                            run_id,
                            int(wm.cycle),
                            len(invalid_projection["missing_agents"]),
                            invalid_projection["missing_agents"][:4],
                        )
                        if self._should_log_cycle_progress(cycle):
                            self._log_cycle_progress(
                                run_id=run_id,
                                cycle=cycle,
                                cycles=cycles,
                                missing_cycles=missing_cycles,
                                invalid_projection_cycles=invalid_projection_cycles,
                                started_at=started_at,
                            )
                        self.servicer.discard_trainer_before(cycle)
                        continue
                    obs_row = self._obs_for_cycle(wm, agent_index)

                cycles.append(int(wm.cycle))
                if should_save_state:
                    offset = append_world_model(states_path, wm)
                    state_offsets.append({"cycle": int(wm.cycle), "offset": offset})
                if should_save_obs and obs_row is not None:
                    obs_rows.append(obs_row)

                self.servicer.discard_trainer_before(cycle)
                if self._should_log_cycle_progress(cycle):
                    self._log_cycle_progress(
                        run_id=run_id,
                        cycle=cycle,
                        cycles=cycles,
                        missing_cycles=missing_cycles,
                        invalid_projection_cycles=invalid_projection_cycles,
                        started_at=started_at,
                    )

            if should_save_obs and not obs_rows and self.config.time_up > 0:
                raise RuntimeError(
                    "No projectable trainer world models collected for obs dataset "
                    f"run_id={run_id} missing_cycles={len(missing_cycles)} "
                    f"invalid_projection_cycles={len(invalid_projection_cycles)}"
                )
            if should_save_obs:
                write_obs(run_dir / "obs.npy", obs_rows)
                logger.warning("Wrote obs array run_id=%s path=%s rows=%d", run_id, run_dir / "obs.npy", len(obs_rows))
            write_cycles(run_dir / "cycles.npy", cycles)
            logger.warning("Wrote cycles array run_id=%s path=%s count=%d", run_id, run_dir / "cycles.npy", len(cycles))
            if state_offsets:
                write_json(run_dir / "state_index.json", state_offsets)
                logger.warning("Wrote state index run_id=%s path=%s entries=%d", run_id, run_dir / "state_index.json", len(state_offsets))

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
                invalid_projection_cycles=invalid_projection_cycles,
            )
            manifest_path = run_dir / "manifest.json"
            write_json(manifest_path, manifest)
            logger.warning(
                "Finished pretrain dataset run run_id=%s manifest=%s collected_cycles=%d "
                "missing_cycles=%d invalid_projection_cycles=%d elapsed_s=%.1f",
                run_id,
                manifest_path,
                len(cycles),
                len(missing_cycles),
                len(invalid_projection_cycles),
                time.monotonic() - started_at,
            )
            return DatasetRunResult(
                run_dir=run_dir,
                manifest_path=manifest_path,
                cycles=cycles,
                missing_cycles=missing_cycles,
                invalid_projection_cycles=invalid_projection_cycles,
            )
        finally:
            if room is not None:
                logger.warning("Releasing dataset room run_id=%s room=%s", run_id, room.info.name)
                room.release()
            if server is not None and loop is not None:
                logger.warning("Stopping dataset gRPC server run_id=%s", run_id)
                self._stop_grpc_server(server, loop)

    def _should_log_cycle_progress(self, cycle: int) -> bool:
        if not self.config.progress.enabled:
            return False
        return (
            cycle == 1
            or cycle == self.config.time_up
            or cycle % self.config.progress.cycle_log_interval == 0
        )

    def _log_cycle_progress(
        self,
        *,
        run_id: str,
        cycle: int,
        cycles: list[int],
        missing_cycles: list[int],
        invalid_projection_cycles: list[dict[str, object]],
        started_at: float,
    ) -> None:
        elapsed_s = max(time.monotonic() - started_at, 0.001)
        progress_pct = (cycle / self.config.time_up * 100.0) if self.config.time_up else 100.0
        logger.warning(
            "Dataset run progress run_id=%s cycle=%d/%d progress=%.1f%% collected=%d "
            "missing=%d invalid_projection=%d cycles_per_s=%.2f",
            run_id,
            cycle,
            self.config.time_up,
            progress_pct,
            len(cycles),
            len(missing_cycles),
            len(invalid_projection_cycles),
            cycle / elapsed_s,
        )

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

    @staticmethod
    def _invalid_projection_cycle(
        wm: pb2.WorldModel,
        agent_index: list[AgentIndexEntry],
    ) -> dict[str, object] | None:
        missing_agents = missing_projectable_agents(wm, agent_index)
        if not missing_agents:
            return None
        return {
            "cycle": int(wm.cycle),
            "reason": "missing_agent_in_trainer_world_model",
            "our_side": int(wm.our_side),
            "our_team_name": wm.our_team_name,
            "their_team_name": wm.their_team_name,
            "our_player_unums": sorted(int(unum) for unum in wm.our_players_dict.keys()),
            "their_player_unums": sorted(int(unum) for unum in wm.their_players_dict.keys()),
            "missing_agents": [entry.to_json() for entry in missing_agents],
        }

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
        invalid_projection_cycles: list[dict[str, object]],
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
                "invalid_projection": invalid_projection_cycles,
            },
        }

    @staticmethod
    def _stop_grpc_server(server: object, loop: asyncio.AbstractEventLoop) -> None:
        async def _shutdown() -> None:
            await server.stop(grace=5)

        future = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
        future.result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
