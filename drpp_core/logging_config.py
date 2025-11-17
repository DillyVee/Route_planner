"""
Logging configuration for DRPP solver.

Provides structured logging with appropriate levels and formatting.
Supports both console and file output with rotation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler


# Default format for log messages
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True,
    detailed: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """Configure logging for DRPP solver.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, logs to console only.
        console: Whether to log to console (stdout)
        detailed: Whether to use detailed format (includes file/line number)
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logging(level=logging.DEBUG, log_file=Path('drpp.log'))
        >>> logger.info("Starting DRPP solver")
    """
    # Get root logger for DRPP
    logger = logging.getLogger("drpp_core")
    logger.setLevel(level)
    logger.handlers.clear()  # Remove any existing handlers

    # Choose format
    fmt = DETAILED_FORMAT if detailed else DEFAULT_FORMAT
    formatter = logging.Formatter(fmt)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.debug("Processing cluster 5")
    """
    return logging.getLogger(f"drpp_core.{name}")


class LogTimer:
    """Context manager for timing operations with automatic logging.

    Example:
        >>> logger = get_logger(__name__)
        >>> with LogTimer(logger, "Dijkstra computation"):
        ...     distances, predecessors = graph.dijkstra(source_id)
        INFO - Dijkstra computation: 0.45s
    """

    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.INFO):
        """Initialize timer.

        Args:
            logger: Logger instance to use
            operation: Description of the operation being timed
            level: Log level for the timing message
        """
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time: Optional[float] = None

    def __enter__(self):
        """Start timing."""
        import time

        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log duration."""
        import time

        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            self.logger.log(self.level, f"{self.operation}: {elapsed:.2f}s")
        return False  # Don't suppress exceptions


# Initialize default logger
_default_logger = setup_logging()


def log_exception(logger: logging.Logger, message: str, exc: Exception) -> None:
    """Log an exception with full traceback.

    Args:
        logger: Logger instance
        message: Context message
        exc: Exception that was caught

    Example:
        >>> try:
        ...     result = risky_operation()
        >>> except ValueError as e:
        ...     log_exception(logger, "Failed to process cluster", e)
    """
    import traceback

    logger.error(f"{message}: {str(exc)}")
    logger.debug(
        f"Traceback:\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
    )
