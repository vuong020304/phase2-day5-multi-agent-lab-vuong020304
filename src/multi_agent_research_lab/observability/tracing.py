"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

logger = logging.getLogger("multi_agent_research_lab.observability")


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context with automated logging of span lifecycles and durations."""
    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    logger.info(f"[Span Start] {name} | Attributes: {attributes or {}}")
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
        logger.info(f"[Span End] {name} | Duration: {span['duration_seconds']:.4f}s")
