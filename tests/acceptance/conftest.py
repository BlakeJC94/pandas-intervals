import time
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    n_intervals,
    expected_ratio,
    results_record,
    n_trials=5,
):
    exp_result = []
    for _ in range(n_trials):
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

    results = pd.DataFrame(exp_result, columns=["time_0", "time_1"])
    ratio = calculate_ratio(results)
    assert ratio > expected_ratio, f"{n_intervals=}, {ratio=}, {expected_ratio=}"
    results_record.append((n_intervals, results))
    return results


def calculate_ratio(results: pd.DataFrame):
    results = results.mean()
    return results["time_0"] / results["time_1"]


@pytest.fixture(scope="class")
def results_record(request):
    file_name = request.cls.__name__.replace(".", "-")
    artifacts_dir = Path("./tests/artifacts")
    for fp in artifacts_dir.glob(f"{file_name}.*"):
        fp.unlink()

    file_path = artifacts_dir / file_name

    results = []
    yield results

    if len(results) > 1:
        results = pd.concat([df.assign(n_intervals=n_ints) for n_ints, df in results])
        results.to_csv(file_path.with_suffix(".csv"), index=False)

        fig = plot_results(results, file_name)
        fig.savefig(file_path.with_suffix(".png"))


def plot_results(results: pd.DataFrame, title: str):
    results = results.groupby("n_intervals").mean().sort_index()

    exps_n_intervals = results.index.tolist()
    x = np.arange(len(exps_n_intervals))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")

    for col in ["time_0", "time_1"]:
        measurement = results[col].to_numpy()
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement * 1e-6, width, label=col)
        ax.bar_label(
            rects,
            labels=[str(timedelta(microseconds=t * 1e-3)) for t in measurement],
            padding=3,
        )
        multiplier += 1

    ax.set_ylabel("Time (ms)")
    ax.set_title(title)
    ax.set_xticks(x + width, exps_n_intervals)
    ax.legend(loc="upper left", ncols=2)

    return fig
