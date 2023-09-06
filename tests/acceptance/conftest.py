import time
import os

import pytest


skipif_pytest_acceptance_not_set = pytest.mark.skipif(
    not os.environ.get("PYTEST_ACCEPTANCE", "False").lower().startswith("t"),
    reason="PYTEST_ACCEPTANCE=True not set in environment",
)


def benchmark(func):
    """Decorator for benchmarking functions.

    Decorated function output is a 2-Tuple where the first element is the inner function return
    value, and the second value is the recorded execution duration.
    """
    def _inner(*args, **kwargs):
        start_time = time.monotonic_ns()
        output = func(*args, **kwargs)
        end_time = time.monotonic_ns()
        return output, end_time - start_time
    return _inner
