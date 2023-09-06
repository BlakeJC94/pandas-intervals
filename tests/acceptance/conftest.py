import time
import os
from copy import deepcopy
from datetime import timedelta

import pytest
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_EXPS_N_INTERVALS = [
    20,
    100,
    500,
    1000,
]
DEFAULT_N_TRIALS_PER_EXP = 3

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


def run(
    arrange_fn,
    act_0_fn,
    act_1_fn,
    check_fn,
    exps_n_intervals=None,
    n_trials_per_exp=None,
):
    exps_n_intervals = exps_n_intervals or DEFAULT_EXPS_N_INTERVALS
    n_trials_per_exp = n_trials_per_exp or DEFAULT_N_TRIALS_PER_EXP

    results = []
    for n_intervals in exps_n_intervals:
        exp_result = []
        for i_trial in range(n_trials_per_exp):
            # logger.debug(f"  {i_trial = }")
            trial_result = []

            args, kwargs = arrange_fn(n_intervals)

            args_0, kwargs_0 = deepcopy(args), deepcopy(kwargs)
            output_0, time_0 = act_0_fn(*args_0, **kwargs_0)
            trial_result.append(time_0)

            args_1, kwargs_1 = deepcopy(args), deepcopy(kwargs)
            output_1, time_1 = act_1_fn(*args_1, **kwargs_1)
            trial_result.append(time_1)

            check_fn(
                output_0,
                output_1,
                *args,
            )

            exp_result.append(trial_result)
        results.append(exp_result)
    return np.array(results)


def plot_results(results, exps_n_intervals=None):
    exps_n_intervals = exps_n_intervals or DEFAULT_EXPS_N_INTERVALS

    x = np.arange(len(exps_n_intervals))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")

    for i, algo in enumerate(["non-vec", "vec"]):
        measurement = results.mean(axis=1)[:, i]
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement * 1e-6, width, label=algo)
        ax.bar_label(
            rects,
            labels=[str(timedelta(microseconds=t * 1e-3)) for t in measurement],
            padding=3,
        )
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Time (ms)")
    ax.set_title("Algo speed by exp (n_intervals)")
    ax.set_xticks(x + width, exps_n_intervals)
    ax.legend(loc="upper left", ncols=3)

    plt.show()
    return fig
