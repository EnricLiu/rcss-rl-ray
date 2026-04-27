from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import gymnasium
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from rcss_env.grpc_srv.batch_queue import BatchQueue, BatchQueueDispatchError
from rcss_env.grpc_srv.proto import pb2
import rcss_env.grpc_srv.servicer as servicer_module
from rcss_env.grpc_srv.servicer import GameServicer
from rcss_env.grpc_srv.truth_buffer import TruthWorldModelTimeoutError


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


def test_game_servicer_fetch_states_timeout_raises_reset_needed_without_shrinking_sync_set() -> None:
	loop = asyncio.new_event_loop()
	started = threading.Event()

	def loop_runner() -> None:
		asyncio.set_event_loop(loop)
		started.set()
		loop.run_forever()

	thread = threading.Thread(target=loop_runner, daemon=True)
	thread.start()
	started.wait(timeout=1.0)

	servicer = GameServicer()
	servicer.register(1)
	servicer.bind_loop(loop)

	try:
		servicer.reset()

		batcher_task = servicer._batcher._BatchQueue__task
		assert batcher_task is not None
		assert not batcher_task.done()

		with pytest.raises(gymnasium.error.ResetNeeded, match="Failed to fetch aligned state"):
			servicer.fetch_states(timeout=0.01)
		assert servicer.unums == frozenset({1})
		snapshot = servicer.debug_snapshot()
		assert snapshot["runtime_error"] == {
			"type": "SyncSetRuntimeError",
			"message": "Failed to fetch aligned state for unum=1: TimeoutError: ",
		}
	finally:
		asyncio.run_coroutine_threadsafe(servicer._batcher.reset(), loop).result(timeout=1.0)
		loop.call_soon_threadsafe(loop.stop)
		thread.join(timeout=1.0)
		loop.close()


def test_game_servicer_action_wait_timeout_raises_reset_needed(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	loop = asyncio.new_event_loop()
	started = threading.Event()

	def loop_runner() -> None:
		asyncio.set_event_loop(loop)
		started.set()
		loop.run_forever()

	thread = threading.Thread(target=loop_runner, daemon=True)
	thread.start()
	started.wait(timeout=1.0)

	servicer = GameServicer()
	servicer.register(1)
	servicer.bind_loop(loop)
	monkeypatch.setattr(servicer_module, "ACTION_GET_TIMEOUT_S", 0.05)

	try:
		servicer.reset()

		state = pb2.State(
			world_model=pb2.WorldModel(
				cycle=7,
				self=pb2.Self(uniform_number=1),
			),
		)
		future = asyncio.run_coroutine_threadsafe(
			servicer.GetPlayerActions(state, None),
			loop,
		)

		with pytest.raises(Exception, match="Timed out waiting for RL actions"):
			future.result(timeout=1.0)

		snapshot = servicer.debug_snapshot()
		assert snapshot["runtime_error"] == {
			"type": "ActionTimeoutError",
			"message": "Timed out waiting for RL actions for unum=1 cycle=7 after 0.05s",
		}

		with pytest.raises(gymnasium.error.ResetNeeded, match="Timed out waiting for RL actions"):
			servicer.send_actions({1: pb2.PlayerActions()})
	finally:
		asyncio.run_coroutine_threadsafe(servicer._batcher.reset(), loop).result(timeout=1.0)
		loop.call_soon_threadsafe(loop.stop)
		thread.join(timeout=1.0)
		loop.close()


def test_game_servicer_buffers_coach_truth_by_exact_cycle() -> None:
	loop = asyncio.new_event_loop()
	started = threading.Event()

	def loop_runner() -> None:
		asyncio.set_event_loop(loop)
		started.set()
		loop.run_forever()

	thread = threading.Thread(target=loop_runner, daemon=True)
	thread.start()
	started.wait(timeout=1.0)

	servicer = GameServicer()
	servicer.bind_loop(loop)

	try:
		servicer.reset()

		coach_state = pb2.State(
			agent_type=pb2.CoachT,
			world_model=pb2.WorldModel(
				cycle=7,
				our_team_score=3,
				game_mode_type=pb2.GameModeType.PlayOn,
			),
		)
		future = asyncio.run_coroutine_threadsafe(
			servicer.GetCoachActions(coach_state, None),
			loop,
		)
		assert future.result(timeout=1.0) == pb2.CoachActions()

		truth = servicer.fetch_truth_world_model(7, timeout=0.1)
		assert truth.cycle == 7
		assert truth.our_team_score == 3

		with pytest.raises(gymnasium.error.ResetNeeded, match="cycle=8") as exc_info:
			servicer.fetch_truth_world_model(8, timeout=0.01)
		assert isinstance(exc_info.value.__cause__, TruthWorldModelTimeoutError)
	finally:
		asyncio.run_coroutine_threadsafe(servicer._truth_buffer.reset(), loop).result(timeout=1.0)
		loop.call_soon_threadsafe(loop.stop)
		thread.join(timeout=1.0)
		loop.close()

