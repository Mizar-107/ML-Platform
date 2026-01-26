"""Structured logging configuration."""

import logging
import sys
from typing import Any

import structlog


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    service_name: str = "llm-mlops",
) -> None:
    """Setup structured logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_type: Output format (json, console)
        service_name: Service name for log context
    """
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Configure processors
    shared_processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.contextvars.merge_contextvars,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if format_type == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Add service context
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(service=service_name)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a configured logger.

    Args:
        name: Logger name (defaults to caller module)

    Returns:
        Structlog BoundLogger instance
    """
    return structlog.get_logger(name)


class LoggerAdapter:
    """Adapter for standard library logger compatibility."""

    def __init__(self, logger: structlog.stdlib.BoundLogger):
        self._logger = logger

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._logger.debug(msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._logger.error(msg, **kwargs)

    def exception(self, msg: str, **kwargs: Any) -> None:
        self._logger.exception(msg, **kwargs)

    def bind(self, **kwargs: Any) -> "LoggerAdapter":
        """Bind additional context to logger."""
        return LoggerAdapter(self._logger.bind(**kwargs))
