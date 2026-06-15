import logging
import time
from contextlib import contextmanager
from typing import Callable, Optional


@contextmanager
def timer(context: str = None, logger_fn: Optional[Callable] = None):
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

    Yields
    ----------
    None
        The context manager does not yield any value; it only measures and prints elapsed time.

    Prints
    ------
    Elapsed time
        The time taken to execute the code block, optionally with the provided context description.
    """
    t = time.perf_counter()

    yield

    time_passed = time.perf_counter() - t

    if context is not None:
        if logger_fn is not None:
            logger_fn(f"{context}: Elapsed time: {time_passed}")
        else:
            print(f"{context}: Elapsed time: {time_passed}")
    else:
        if logger_fn is not None:
            logger_fn(f"Elapsed time: {time_passed}")
        else:
            print(f"Elapsed time:{time_passed} ")
