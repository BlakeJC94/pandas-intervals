from tests.acceptance.conftest import benchmark, run, skipif_pytest_acceptance_not_set
from tests.conftest import assert_df_interval_times_equal, random_intervals


def arrange(n_intervals):
    df_a = random_intervals(n_intervals=n_intervals).ivl()
    df_b = random_intervals(n_intervals=n_intervals).ivl()
    return [df_a, df_b], {}


@benchmark
def act_0(df_a, df_b):
    return df_a.ivl.basic.truncate(df_b)


@benchmark
def act_1(df_a, df_b):
    return df_a.ivl.truncate(df_b)


@skipif_pytest_acceptance_not_set
def test_truncate():
    exps_n_intervals = [
        20,
        100,
        500,
        1000,
        2000,
    ]
    results = run(
        arrange,
        act_0,
        act_1,
        assert_df_interval_times_equal,
        exps_n_intervals=exps_n_intervals,
    )
    # plot_results(results, exps_n_intervals)
    # assert False
