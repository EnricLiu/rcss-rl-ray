from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from rcss_env.grpc_srv.batch_queue import BatchQueue
from rcss_env.grpc_srv.servicer import GameServicer


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


def test_game_servicer_fetch_states_timeout_returns_empty_snapshot() -> None:
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

		assert servicer.fetch_states(timeout=0.01) == {}
		assert servicer.unums == frozenset()
	finally:
		asyncio.run_coroutine_threadsafe(servicer._batcher.reset(), loop).result(timeout=1.0)
		loop.call_soon_threadsafe(loop.stop)
		thread.join(timeout=1.0)
		loop.close()

