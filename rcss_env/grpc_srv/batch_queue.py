"""Batched async state queue.

Collects per-unum State messages arriving from concurrent gRPC streams,
waits until all registered unums have reported for a given timestep,
then dispatches the complete batch to each unum's individual output queue.
"""

from asyncio import Queue, Task

import asyncio
import logging

QUEUE_SEND_TIMEOUT_S = 2.0
RESET_TIMEOUT_S = 5.0


class BatchQueue[StateTy]:
    """Timestep-synchronised state batcher for multiple unums.

    Generic over *StateTy* (typically ``pb2.State``).  Incoming states are
    indexed by ``(timestep, unum)``; once every registered unum has reported
    for a timestep the whole batch is dispatched via per-unum output queues.
    """

    def __init__(self, unums: set[int] | None = None, queue_send_timeout_s: float = QUEUE_SEND_TIMEOUT_S, reset_timeout_s: float = RESET_TIMEOUT_S):
        self.__unums = unums or set()

        # timestep -> unum -> state
        self.__states: dict[int, dict[int, StateTy]] = {}

        # unum -> output queue of (timestep, state)
        self.__queues: dict[int, Queue[tuple[int, StateTy]]] = {}

        self.__reset_event: Queue[bool] = Queue(maxsize=1)
        self.__update_event: Queue[bool] = Queue(maxsize=1)

        self.__last_timestep: int = -1

        self.__reset_timeout_s = reset_timeout_s
        self.__queue_send_timeout_s = queue_send_timeout_s

        self.__task: Task | None = None

    async def reset(self):
        """Signal the running dispatch loop to stop and clear all internal state."""
        logging.warning(
            "BatchQueue.reset: stopping dispatch task (task_done=%s), registered_unums=%s, pending_timesteps=%s",
            self.__task.done() if self.__task else "no-task",
            sorted(self.__unums),
            sorted(self.__states.keys()),
        )
        if self.__task is not None and not self.__task.done():
            await self.__reset_event.put(True)
            try: await asyncio.wait_for(self.__task, timeout=self.__reset_timeout_s)
            except asyncio.TimeoutError:
                logging.warning("BatchQueue.reset: timeout waiting for dispatch task to finish; cancelling")
                self.__task.cancel()
        self.__task = None

        self.__unums.clear()
        self.__states.clear()
        self.__queues.clear()
        self.__reset_event = Queue(maxsize=1)
        self.__update_event = Queue(maxsize=1)
        self.__last_timestep = -1
        logging.warning("BatchQueue.reset: complete")

    def register(self, unum: int, queue: Queue[tuple[int, StateTy]]):
        """Register a unum and its output queue for batch dispatch."""
        self.__unums.add(unum)
        self.__queues[unum] = queue
        logging.warning("BatchQueue.register: unum=%d registered, all_unums=%s", unum, sorted(self.__unums))

    def unregister(self, unum: int) -> Queue[tuple[int, StateTy]] | None:
        """Remove a unum from the batch set and return its queue (if any)."""
        self.__unums.discard(unum)
        logging.warning("BatchQueue.unregister: unum=%d removed, remaining_unums=%s", unum, sorted(self.__unums))
        return self.__queues.pop(unum, None)

    def unums(self) -> frozenset[int]:
        """Return the current set of registered unums."""
        return frozenset(self.__unums)

    def has(self, unum: int) -> bool:
        """Check whether a unum is currently registered."""
        return unum in self.__unums

    def snapshot(self) -> dict[str, object]:
        """Return a structured view of the batcher's buffered state for debugging."""
        queued_timesteps: dict[str, dict[str, object]] = {}
        for timestep in sorted(self.__states.keys()):
            arrived_unums = set(self.__states[timestep].keys())
            queued_timesteps[str(timestep)] = {
                "arrived_unums": sorted(arrived_unums),
                "missing_unums": sorted(self.__unums - arrived_unums),
            }

        return {
            "registered_unums": sorted(self.__unums),
            "last_timestep": self.__last_timestep,
            "pending_timesteps": sorted(self.__states.keys()),
            "queued_timesteps": queued_timesteps,
            "output_queue_sizes": {
                str(unum): queue.qsize()
                for unum, queue in sorted(self.__queues.items())
            },
            "dispatch_task": {
                "exists": self.__task is not None,
                "running": self.__task is not None and not self.__task.done(),
                "done": self.__task.done() if self.__task is not None else None,
                "cancelled": self.__task.cancelled() if self.__task is not None else None,
            },
        }

    async def __run(self):
        """Main dispatch loop: wait for updates or reset, then dispatch complete batches."""
        logging.warning("BatchQueue.__run: dispatch loop started, registered_unums=%s", sorted(self.__unums))
        try:
            while True:
                reset_task = asyncio.ensure_future(self.__reset_event.get())
                update_task = asyncio.ensure_future(self.__update_event.get())

                done, pending = await asyncio.wait(
                    (reset_task, update_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending: task.cancel()
                if reset_task in done:
                    logging.warning("BatchQueue.__run: received reset event, exiting loop")
                    break

                # Process buffered timesteps in order
                pending_timesteps = sorted(self.__states.keys())
                logging.warning(
                    "BatchQueue.__run: update event received, pending_timesteps=%s, last_timestep=%d, registered_unums=%s",
                    pending_timesteps, self.__last_timestep, sorted(self.__unums),
                )
                for timestep in pending_timesteps:
                    # Discard stale timesteps
                    if timestep <= self.__last_timestep:
                        logging.warning(
                            "BatchQueue.__run: discarding stale timestep=%d (last_timestep=%d)",
                            timestep, self.__last_timestep,
                        )
                        self.__states.pop(timestep)
                        continue

                    arrived_unums = set(self.__states[timestep].keys())
                    missing_unums = self.__unums - arrived_unums
                    # Skip if not all registered unums have reported yet
                    if missing_unums:
                        logging.warning(
                            "BatchQueue.__run: timestep=%d incomplete — arrived=%s, missing=%s",
                            timestep, sorted(arrived_unums), sorted(missing_unums),
                        )
                        continue

                    self.__last_timestep = timestep
                    logging.warning(
                        "BatchQueue.__run: dispatching complete batch for timestep=%d, unums=%s",
                        timestep, sorted(arrived_unums),
                    )

                    states: dict[int, StateTy] = self.__states.pop(timestep)
                    unums = states.keys()
                    tasks = [
                        asyncio.wait_for(self.__queues[unum].put((timestep, s)), timeout=self.__queue_send_timeout_s)
                        for unum, s in states.items()
                    ]

                    res = await asyncio.gather(*tasks, return_exceptions=True)
                    for unum, r in zip(unums, res):
                        if r is not None:
                            logging.error(
                                "BatchQueue.__run: timeout sending state for unum=%d at timestep=%d; unregistering",
                                unum, timestep,
                            )
                            self.unregister(unum)
                    break

        except Exception as e:
            logging.error("BatchQueue.__run: dispatch loop crashed: %s", e, exc_info=True)

        logging.warning("BatchQueue.__run: dispatch loop exited")

    def run(self):
        """Start the async dispatch loop as an asyncio task."""
        if self.__task is not None and not self.__task.done():
            raise RuntimeError("BatchQueue is already running")

        self.__task = None
        self.__task = asyncio.create_task(self.__run())
        self.__task.add_done_callback(self.__on_task_done)
        logging.warning("BatchQueue.run: dispatch task created, registered_unums=%s", sorted(self.__unums))

    def __on_task_done(self, task: "asyncio.Task") -> None:
        """Callback invoked when the dispatch task exits for any reason."""
        if task.cancelled():
            logging.warning("BatchQueue: dispatch task was cancelled")
        elif task.exception() is not None:
            logging.error(
                "BatchQueue: dispatch task exited with exception: %s",
                task.exception(), exc_info=task.exception(),
            )
        else:
            logging.warning("BatchQueue: dispatch task finished normally")

    async def put(self, unum: int, timestep: int, unum_state: StateTy):
        """Submit a state for a (unum, timestep) pair and notify the dispatch loop."""
        if timestep <= self.__last_timestep:
            logging.warning(
                "BatchQueue.put: unum=%d sent state for timestep=%d but last_dispatched=%d; ignoring",
                unum, timestep, self.__last_timestep,
            )
            return

        self.__states[timestep] = self.__states.get(timestep, {})

        if unum in self.__states[timestep]:
            logging.warning(
                "BatchQueue.put: duplicate state for unum=%d at timestep=%d; overwriting",
                unum, timestep,
            )

        self.__states[timestep][unum] = unum_state
        arrived = set(self.__states[timestep].keys())
        logging.warning(
            "BatchQueue.put: unum=%d timestep=%d stored — arrived=%s, waiting_for=%s, task_alive=%s",
            unum, timestep, sorted(arrived), sorted(self.__unums - arrived),
            self.__task is not None and not self.__task.done(),
        )

        # Warn if the dispatch task is not alive when a state arrives
        if self.__task is None or self.__task.done():
            logging.error(
                "BatchQueue.put: unum=%d timestep=%d — dispatch task is NOT running (task=%s)! "
                "States will never be dispatched.",
                unum, timestep, self.__task,
            )

        try:
            self.__update_event.put_nowait(True)
        except asyncio.QueueFull:
            logging.warning("BatchQueue.put: update event queue is full (signal already pending)")
