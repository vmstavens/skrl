import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Callable, Dict, List, Optional


class TimerManager:
    """Class to manage multiple timers and handle data storage"""

    _timings: Dict[str, List[float]] = {}
    _enabled = True

    @classmethod
    def record_timing(cls, context: str, elapsed_time: float):
        """Record a timing measurement"""
        if not cls._enabled:
            return

        if context not in cls._timings:
            cls._timings[context] = []
        cls._timings[context].append(elapsed_time)

    @classmethod
    def get_timings(cls) -> Dict[str, List[float]]:
        """Get all recorded timings"""
        return cls._timings.copy()

    @classmethod
    def clear_timings(cls):
        """Clear all recorded timings"""
        cls._timings.clear()

    @classmethod
    def enable(cls):
        """Enable timing recording"""
        cls._enabled = True

    @classmethod
    def disable(cls):
        """Disable timing recording"""
        cls._enabled = False

    @classmethod
    def save_to_json(cls, filename: Optional[str] = None, include_stats: bool = True):
        """Save timings to JSON file"""
        if not cls._timings:
            print("No timing data to save")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"timings_{timestamp}.json"

        # Create data structure
        data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "total_measurements": sum(
                    len(times) for times in cls._timings.values()
                ),
                "unique_contexts": len(cls._timings),
            },
            "timings": cls._timings,
        }

        # Add statistics if requested
        if include_stats:
            data["statistics"] = {}
            for context, times in cls._timings.items():
                data["statistics"][context] = {
                    "count": len(times),
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "total": sum(times),
                }

        # Ensure directory exists
        os.makedirs(
            os.path.dirname(filename) if os.path.dirname(filename) else ".",
            exist_ok=True,
        )

        # Save to file
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Timings saved to {filename}")
        return filename


@contextmanager
def timer(
    context: str = None,
    logger_fn: Optional[Callable] = None,
    record: bool = True,
    log: bool = False,
):
    """
    Context manager to measure and print the elapsed time of a code block.

    This context manager allows for the measurement of execution time within a code block.
    It prints the elapsed time upon exiting the context, optionally with a provided
    description.

    Parameters
    ----------
    context : str, optional
        A description to include in the printout alongside the elapsed time. If not provided,
        only the elapsed time will be printed.
    logger_fn : Callable, optional
        A logging function to use instead of print (e.g., logging.info)
    record : bool, default True
        Whether to record this timing in the TimerManager for later export

    Yields
    ------
    None
        The context manager does not yield any value; it only measures and prints elapsed time.
    """
    t = time.perf_counter()
    yield
    time_passed = time.perf_counter() - t

    # Record timing if enabled
    if record and context is not None:
        TimerManager.record_timing(context, time_passed)

    if not log:
        return

    # Log/print the result
    if context is not None:
        message = f"{context}: Elapsed time: {time_passed}"
    else:
        message = f"Elapsed time: {time_passed}"

    if logger_fn is not None:
        logger_fn(message)
    else:
        print(message)


# Convenience functions for easy access
def save_timings(filename: Optional[str] = None, include_stats: bool = True):
    """Convenience function to save all recorded timings"""
    return TimerManager.save_to_json(filename, include_stats)


def clear_timings():
    """Convenience function to clear all recorded timings"""
    TimerManager.clear_timings()


def get_timings() -> Dict[str, List[float]]:
    """Convenience function to get all recorded timings"""
    return TimerManager.get_timings()
