"""Command-line entry point for RCSS model inference."""

from __future__ import annotations

import argparse
import json
import logging
import signal
from pathlib import Path

from pydantic import ValidationError

from .config import InferenceConfig, load_inference_config
from .loader import BundleValidationError, ModelLoadError
from .policy import PolicyInferenceError
from .runner import RunnerInfrastructureError, execute_inference


LoggingState = tuple[list[logging.Handler], int]


def configure_logging(level: str) -> LoggingState:
    root = logging.getLogger()
    previous = (list(root.handlers), root.level)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root.handlers = [handler]
    root.setLevel(getattr(logging, level))
    return previous


def restore_logging(state: LoggingState) -> None:
    root = logging.getLogger()
    for handler in root.handlers:
        if handler not in state[0]:
            handler.close()
    root.handlers = state[0]
    root.setLevel(state[1])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RCSS MultiRLModule inference")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--bundle")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--episodes", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--deterministic", action="store_true", default=None)
    mode.add_argument("--stochastic", dest="deterministic", action="store_false")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stop_state = {"requested": False, "signal": None}

    def _handle_signal(signum: int, frame) -> None:
        stop_state["requested"] = True
        stop_state["signal"] = signum

    previous_handlers = {
        signum: signal.signal(signum, _handle_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    logging_state: LoggingState | None = None
    try:
        config = load_inference_config(args.config)
        updates = {}
        for key in ("bundle", "device", "episodes", "deterministic"):
            value = getattr(args, key)
            if value is not None:
                updates["bundle_path" if key == "bundle" else key] = value
        if updates:
            config = InferenceConfig.model_validate(
                config.model_dump() | updates
            )

        logging_state = configure_logging(config.logging.level)
        summary = execute_inference(
            config,
            stop_requested=lambda: bool(stop_state["requested"]),
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        if stop_state["signal"] == signal.SIGINT:
            return 130
        if stop_state["signal"] == signal.SIGTERM:
            return 143
        return 0
    except ModelLoadError as exc:
        logging.getLogger(__name__).error("Model load error: %s", exc)
        return 4
    except (ValidationError, ValueError, BundleValidationError) as exc:
        logging.getLogger(__name__).error("Configuration or bundle error: %s", exc)
        return 2
    except RunnerInfrastructureError as exc:
        logging.getLogger(__name__).error("Inference infrastructure error: %s", exc)
        return 3
    except PolicyInferenceError as exc:
        logging.getLogger(__name__).error("Model inference error: %s", exc)
        return 4
    finally:
        if logging_state is not None:
            restore_logging(logging_state)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
