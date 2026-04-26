import time
import logging
from functools import wraps

def retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    logger: logging.Logger = logging.getLogger(__name__),
    exceptions: tuple[type[BaseException], ...] = (Exception,),
):
    """Retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        delay: Initial delay in seconds between retries.
        backoff: Multiplier applied to delay after each retry.
        logger: Logger instance for logging retry attempts.
        exceptions: Tuple of exception types that trigger a retry.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception: BaseException | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(
                        "Attempt %d/%d for %s failed: %s. Retrying in %.1fs...",
                        attempt, max_retries, func.__name__, e, current_delay,
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator
