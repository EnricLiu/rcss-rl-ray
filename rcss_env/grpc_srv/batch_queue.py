from typing import AsyncGenerator, Tuple
from asyncio import Queue, Task

import asyncio
import logging

QUEUE_SEND_TIMEOUT_S = 2.0
RESET_TIMEOUT_S = 5.0

class BatchQueue[StateTy]:
    def __init__(self, unums: set[int] | None = None, queue_send_timeout_s: float = QUEUE_SEND_TIMEOUT_S, reset_timeout_s: float = RESET_TIMEOUT_S):
        self.__unums = unums or set()

        # timestep -> unum -> state
        self.__states: dict[int, dict[int, StateTy]] = {}

        # unum -> queue[ts, state]
        self.__queues: dict[int, Queue[tuple[int, StateTy]]] = {}

        self.__reset_event: Queue[bool] = Queue(maxsize=1)
        self.__update_event: Queue[bool] = Queue(maxsize=1)

        self.__last_timestep: int = -1

        self.__reset_timeout_s = reset_timeout_s
        self.__queue_send_timeout_s = queue_send_timeout_s

        self.__task: Task | None = None


    async def reset(self):
        if self.__task is not None and not self.__task.done():
            await self.__reset_event.put(True)
            try: await asyncio.wait_for(self.__task, timeout=self.__reset_timeout_s)
            except asyncio.TimeoutError:
                logging.warning("Timeout while waiting for reset event acknowledgment")
            finally:
                self.__task = None

        self.__unums.clear()
        self.__states.clear()
        self.__reset_event = Queue(maxsize=1)
        self.__update_event = Queue(maxsize=1)
        self.__last_timestep = -1

    def register(self, unum: int, queue: Queue[tuple[int, StateTy]]):
        self.__unums.add(unum)
        self.__queues[unum] = queue

    def unregister(self, unum: int) -> Queue[tuple[int, StateTy]] | None:
        self.__unums.remove(unum)
        return self.__queues.pop(unum, None)

    def unums(self) -> frozenset[int]:
        return frozenset(self.__unums)

    def has(self, unum: int) -> bool:
        return unum in self.__unums

    async def __run(self):
        """
        get a receiver generator that yields (timestep, {unum: state})
        whenever a new batch of states is available for all unums at a given timestep.

        The generator will yield batches in order of timestep,
        and will skip any batches for timesteps that are less than or equal to the last yielded timestep.
        Returns:
            AsyncGenerator[Tuple[int, dict[int, StateTy]], None]: An async generator that yields (timestep, {unum: state}) tuples.
        """
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
                    logging.info("Receiver received reset event")
                    break

                for timestep in sorted(self.__states.keys()):
                    if timestep <= self.__last_timestep:
                        self.__states.pop(timestep)
                        continue

                    if self.__unums != set(self.__states[timestep].keys()):
                        continue

                    self.__last_timestep = timestep

                    states: dict[int, StateTy] = self.__states.pop(timestep) # unum, state
                    unums = states.keys()
                    tasks = [
                        asyncio.wait_for(self.__queues[unum].put((timestep, s)), timeout=self.__queue_send_timeout_s)
                        for unum, s in states.items()
                    ]

                    res = await asyncio.gather(*tasks, return_exceptions=True)
                    for unum, r in zip(unums, res):
                        if r is not None:
                            logging.error(f"Timeout while sending state for unum={unum} at timestep={timestep}")
                            self.unregister(unum)
                            logging.error("Unregistered unum {unum} due to send timeout")
                    break

        except Exception as e:
            logging.error(f"Receiver encountered exception and broke: {e}")

    def run(self):
        if self.__task is not None:
            raise RuntimeError("BatchQueue is already running")

        self.__task = asyncio.create_task(self.__run())

    async def put(self, unum: int, timestep: int, unum_state: StateTy):
        if timestep <= self.__last_timestep:
            logging.warning(f"Received state for {timestep}ts but already finished {self.__last_timestep}ts; ignoring")
            return

        self.__states[timestep] = self.__states.get(timestep, {})

        if unum in self.__states[timestep]:
            logging.warning(f"Received duplicate state for unum={unum} at timestep={timestep}; overwriting")

        self.__states[timestep][unum] = unum_state

        try:
            self.__update_event.put_nowait(True)
        except asyncio.QueueFull:
            logging.info("Update event queue is full")
