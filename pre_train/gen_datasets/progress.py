from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from .config import DatasetProgressConfig, TqdmMode

T = TypeVar("T")


def _tqdm_disabled(progress: DatasetProgressConfig) -> bool:
    if not progress.enabled or progress.tqdm == TqdmMode.NEVER:
        return True
    if progress.tqdm == TqdmMode.AUTO:
        return not sys.stderr.isatty()
    return False


def iter_progress(
    iterable: Iterable[T],
    *,
    progress: DatasetProgressConfig,
    total: int | None,
    desc: str,
    unit: str,
) -> Iterator[T]:
    if _tqdm_disabled(progress):
        return iter(iterable)
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iter(iterable)
    return iter(
        tqdm(
            iterable,
            total=total,
            desc=desc,
            unit=unit,
            leave=progress.tqdm_leave,
        )
    )


@contextmanager
def manual_progress(
    *,
    progress: DatasetProgressConfig,
    total: int,
    desc: str,
    unit: str,
) -> Iterator[Any | None]:
    if _tqdm_disabled(progress):
        yield None
        return
    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield None
        return
    bar = tqdm(total=total, desc=desc, unit=unit, leave=progress.tqdm_leave)
    try:
        yield bar
    finally:
        bar.close()
