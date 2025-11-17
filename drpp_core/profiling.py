"""
Profiling utilities for performance analysis and optimization.

Provides decorators and context managers for timing, memory profiling,
and identifying bottlenecks.
"""

import time
import functools
from typing import Callable, Optional, Any, Dict
from contextlib import contextmanager
import cProfile
import pstats
import io
from pathlib import Path

from .logging_config import get_logger

logger = get_logger(__name__)

# Try to import memory_profiler
try:
    from memory_profiler import profile as memory_profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    logger.debug("memory_profiler not available")


def time_function(func: Callable) -> Callable:
    """Decorator to time function execution.

    Args:
        func: Function to time

    Returns:
        Wrapped function that logs execution time

    Example:
        >>> @time_function
        ... def expensive_operation():
        ...     # do work
        ...     pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            logger.info(f"{func.__name__} took {elapsed:.3f}s")

    return wrapper


def time_function_detailed(
    log_args: bool = False,
    log_result: bool = False
) -> Callable:
    """Decorator to time function with detailed logging.

    Args:
        log_args: Whether to log function arguments
        log_result: Whether to log function result

    Returns:
        Decorator function

    Example:
        >>> @time_function_detailed(log_args=True)
        ... def process_cluster(cluster_id, segments):
        ...     # do work
        ...     return result
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()

            if log_args:
                logger.debug(f"{func.__name__} called with args={args}, kwargs={kwargs}")

            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start

                if log_result:
                    logger.info(
                        f"{func.__name__} completed in {elapsed:.3f}s, result={result}"
                    )
                else:
                    logger.info(f"{func.__name__} completed in {elapsed:.3f}s")

                return result

            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(
                    f"{func.__name__} failed after {elapsed:.3f}s: {e}",
                    exc_info=True
                )
                raise

        return wrapper

    return decorator


class PerformanceTimer:
    """Context manager and decorator for timing code blocks.

    Can be used as both a context manager and a decorator.

    Example:
        >>> # As context manager
        >>> with PerformanceTimer("dijkstra") as timer:
        ...     distances, preds = graph.dijkstra(source)
        >>> print(f"Took {timer.elapsed:.2f}s")

        >>> # As decorator
        >>> @PerformanceTimer("compute_matrix")
        ... def compute_distance_matrix(...):
        ...     pass
    """

    def __init__(self, operation_name: str, log_level: int = 20):  # 20 = INFO
        """Initialize timer.

        Args:
            operation_name: Name of operation being timed
            log_level: Logging level for the timing message
        """
        self.operation_name = operation_name
        self.log_level = log_level
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log."""
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time
            logger.log(
                self.log_level,
                f"{self.operation_name}: {self.elapsed:.3f}s"
            )
        return False

    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


class ProfilerContext:
    """Context manager for cProfile profiling.

    Example:
        >>> with ProfilerContext("distance_matrix", top_n=20):
        ...     matrix = compute_distance_matrix(graph, nodes, coords)
        # Automatically prints top 20 functions by cumulative time
    """

    def __init__(
        self,
        operation_name: str,
        top_n: int = 20,
        sort_by: str = 'cumulative',
        output_file: Optional[Path] = None
    ):
        """Initialize profiler.

        Args:
            operation_name: Name of operation being profiled
            top_n: Number of top functions to display
            sort_by: Sort key ('cumulative', 'time', 'calls')
            output_file: Optional file to save full stats
        """
        self.operation_name = operation_name
        self.top_n = top_n
        self.sort_by = sort_by
        self.output_file = output_file
        self.profiler = cProfile.Profile()

    def __enter__(self):
        """Start profiling."""
        logger.info(f"Starting profiler: {self.operation_name}")
        self.profiler.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and print results."""
        self.profiler.disable()

        # Create stats
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats(self.sort_by)

        # Print summary
        logger.info(f"Profiler results for {self.operation_name}:")
        logger.info(f"Top {self.top_n} functions by {self.sort_by}:")
        ps.print_stats(self.top_n)
        logger.info(s.getvalue())

        # Save to file if requested
        if self.output_file:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            ps.dump_stats(str(self.output_file))
            logger.info(f"Saved profiler stats to {self.output_file}")

        return False


@contextmanager
def track_memory(operation_name: str):
    """Context manager to track memory usage.

    Requires psutil to be installed.

    Args:
        operation_name: Name of operation being tracked

    Example:
        >>> with track_memory("matrix_computation"):
        ...     matrix = compute_distance_matrix(graph, nodes, coords)
        INFO - matrix_computation: 125.3 MB increase
    """
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        yield

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before

        logger.info(
            f"{operation_name}: {mem_increase:+.1f} MB "
            f"(before: {mem_before:.1f} MB, after: {mem_after:.1f} MB)"
        )

    except ImportError:
        logger.warning("psutil not available, cannot track memory")
        yield


class BenchmarkRunner:
    """Run and compare multiple benchmark scenarios.

    Example:
        >>> bench = BenchmarkRunner()
        >>> bench.add_scenario("v3", lambda: route_v3(graph, edges, clusters))
        >>> bench.add_scenario("v4", lambda: route_v4(graph, edges, clusters))
        >>> bench.run(iterations=5)
        >>> bench.print_results()
    """

    def __init__(self):
        """Initialize benchmark runner."""
        self.scenarios: Dict[str, Callable] = {}
        self.results: Dict[str, list] = {}

    def add_scenario(self, name: str, func: Callable[[], Any]) -> None:
        """Add a benchmark scenario.

        Args:
            name: Scenario name
            func: Callable that performs the operation to benchmark
        """
        self.scenarios[name] = func
        logger.debug(f"Added benchmark scenario: {name}")

    def run(self, iterations: int = 3, warmup: int = 1) -> None:
        """Run all scenarios.

        Args:
            iterations: Number of iterations per scenario
            warmup: Number of warmup iterations (not counted)
        """
        logger.info(f"Running benchmarks: {iterations} iterations, {warmup} warmup")

        for name, func in self.scenarios.items():
            logger.info(f"Benchmarking: {name}")
            times = []

            # Warmup
            for i in range(warmup):
                logger.debug(f"  Warmup {i + 1}/{warmup}")
                func()

            # Actual runs
            for i in range(iterations):
                logger.debug(f"  Iteration {i + 1}/{iterations}")
                start = time.perf_counter()
                func()
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                logger.debug(f"  -> {elapsed:.3f}s")

            self.results[name] = times

    def print_results(self) -> None:
        """Print benchmark results."""
        if not self.results:
            logger.warning("No results to print")
            return

        logger.info("=" * 60)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 60)

        for name, times in self.results.items():
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            logger.info(f"{name}:")
            logger.info(f"  Average: {avg_time:.3f}s")
            logger.info(f"  Min:     {min_time:.3f}s")
            logger.info(f"  Max:     {max_time:.3f}s")
            logger.info(f"  Runs:    {times}")

        # Compare scenarios
        if len(self.results) > 1:
            logger.info("")
            logger.info("COMPARISONS:")
            baseline_name = list(self.results.keys())[0]
            baseline_avg = sum(self.results[baseline_name]) / len(self.results[baseline_name])

            for name, times in list(self.results.items())[1:]:
                avg_time = sum(times) / len(times)
                speedup = baseline_avg / avg_time
                logger.info(
                    f"  {name} vs {baseline_name}: {speedup:.2f}x "
                    f"({'faster' if speedup > 1 else 'slower'})"
                )

        logger.info("=" * 60)
