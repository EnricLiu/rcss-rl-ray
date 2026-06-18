from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import gymnasium
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from rcss_env.grpc_srv.batch_queue import BatchQueue, BatchQueueDispatchError
from rcss_env.grpc_srv.proto import pb2
import rcss_env.grpc_srv.servicer as servicer_module
from rcss_env.grpc_srv.servicer import GameServicer, SyncSetRuntimeError
from rcss_env.grpc_srv.truth_buffer import TruthWorldModelTimeoutError


def _register_response(
	agent_type: int = pb2.PlayerT,
	uniform_number: int = 1,
	client_id: int = 1,
) -> pb2.RegisterResponse:
	return pb2.RegisterResponse(
		client_id=client_id,
		agent_type=agent_type,
		team_name="HELIOS",
		uniform_number=uniform_number,
		rpc_server_language_type=pb2.PYThON,
	)


def test_batch_queue_reset_clears_state_and_allows_restart() -> None:
	async def scenario() -> None:
		batcher: BatchQueue[int] = BatchQueue()
		first_queue: asyncio.Queue[tuple[int, int]] = asyncio.Queue(maxsize=1)

		batcher.register(1, first_queue)
		batcher.run()
		await batcher.put(1, 3, 7)

		assert await asyncio.wait_for(first_queue.get(), timeout=0.5) == (3, 7)

		await batcher.reset()
		assert batcher.unums() == frozenset()

		second_queue: asyncio.Queue[tuple[int, int]] = asyncio.Queue(maxsize=1)
		batcher.register(1, second_queue)
		batcher.run()
		await batcher.put(1, 4, 8)

		assert await asyncio.wait_for(second_queue.get(), timeout=0.5) == (4, 8)

		await batcher.reset()

	asyncio.run(scenario())


def test_batch_queue_dispatch_failure_preserves_registered_unums() -> None:
	async def scenario() -> None:
		batcher: BatchQueue[int] = BatchQueue(queue_send_timeout_s=0.01)
		blocked_queue: asyncio.Queue[tuple[int, int]] = asyncio.Queue(maxsize=1)
		await blocked_queue.put((0, 999))

		batcher.register(1, blocked_queue)
		batcher.run()

		await batcher.put(1, 3, 7)
		await asyncio.sleep(0.05)

		assert batcher.unums() == frozenset({1})
		with pytest.raises(BatchQueueDispatchError, match="failed to dispatch state"):
			batcher.raise_if_failed()

		await batcher.reset()

	asyncio.run(scenario())


def test_game_servicer_register_bye_and_planner_rpc() -> None:
	async def scenario() -> None:
		servicer = GameServicer()
		servicer.register(1)

		response = await servicer.Register(
			pb2.RegisterRequest(
				agent_type=pb2.PlayerT,
				team_name="HELIOS",
				uniform_number=1,
				rpc_version=2,
			),
			None,
		)
		assert response.client_id == 1
		assert response.uniform_number == 1
		assert response.rpc_server_language_type == pb2.PYThON

		planner = await servicer.GetBestPlannerAction(
			pb2.BestPlannerActionRequest(
				register_response=response,
				pairs={
					3: pb2.RpcActionState(),
					1: pb2.RpcActionState(),
				},
			),
			None,
		)
		assert planner.index == 1

		assert await servicer.SendByeCommand(response, None) == pb2.Empty()
		snapshot = servicer.debug_snapshot()
		assert snapshot["clients"]["1"]["status"] == "disconnected"
		assert snapshot["last_bye"]["1"]["uniform_number"] == 1

	asyncio.run(scenario())


def test_game_servicer_fetch_states_timeout_raises_reset_needed_without_shrinking_sync_set() -> None:
	async def scenario() -> None:
		servicer = GameServicer()
		servicer.register(1)
		await servicer._GameServicer__reset_runtime()

		batcher_task = servicer._batcher._BatchQueue__task
		assert batcher_task is not None
		assert not batcher_task.done()

		with pytest.raises(SyncSetRuntimeError, match="Failed to fetch aligned state"):
			await servicer._GameServicer__fetch_states(timeout=0.01)
		assert servicer.unums == frozenset({1})
		snapshot = servicer.debug_snapshot()
		assert snapshot["runtime_error"] == {
			"type": "SyncSetRuntimeError",
			"message": "Failed to fetch aligned state for unum=1: TimeoutError: ",
		}
		await servicer._batcher.reset()

	asyncio.run(scenario())


def test_game_servicer_action_wait_timeout_raises_reset_needed(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	async def scenario() -> None:
		servicer = GameServicer()
		servicer.register(1)
		monkeypatch.setattr(servicer_module, "ACTION_GET_TIMEOUT_S", 0.05)
		await servicer._GameServicer__reset_runtime()

		state = pb2.State(
			register_response=_register_response(),
			world_model=pb2.WorldModel(
				cycle=7,
				self=pb2.Self(uniform_number=1),
			),
		)

		with pytest.raises(Exception, match="Timed out waiting for RL actions"):
			await servicer.GetPlayerActions(state, None)

		snapshot = servicer.debug_snapshot()
		assert snapshot["runtime_error"] == {
			"type": "ActionTimeoutError",
			"message": "Timed out waiting for RL actions for unum=1 cycle=7 after 0.05s",
		}

		with pytest.raises(gymnasium.error.ResetNeeded, match="Timed out waiting for RL actions"):
			servicer._raise_if_runtime_error()
		await servicer._batcher.reset()

	asyncio.run(scenario())


def test_game_servicer_buffers_coach_truth_by_exact_cycle() -> None:
	async def scenario() -> None:
		servicer = GameServicer()
		await servicer._GameServicer__reset_runtime()

		coach_state = pb2.State(
			register_response=_register_response(
				agent_type=pb2.CoachT,
				uniform_number=12,
				client_id=12,
			),
			world_model=pb2.WorldModel(
				cycle=7,
				our_team_score=3,
				game_mode_type=pb2.GameModeType.PlayOn,
			),
		)
		assert await servicer.GetCoachActions(coach_state, None) == pb2.CoachActions()

		truth = await servicer._GameServicer__fetch_truth_world_model(7, timeout=0.1)
		assert truth.cycle == 7
		assert truth.our_team_score == 3

		with pytest.raises(TruthWorldModelTimeoutError, match="cycle=8"):
			await servicer._GameServicer__fetch_truth_world_model(8, timeout=0.01)
		await servicer._truth_buffer.reset()

	asyncio.run(scenario())

