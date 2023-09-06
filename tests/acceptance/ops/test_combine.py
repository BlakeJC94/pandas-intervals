from tests.acceptance.conftest import benchmark, run, skipif_pytest_acceptance_not_set
from tests.conftest import assert_df_interval_times_equal, random_intervals

def arrange(n_intervals):
    df_a = random_intervals(n_intervals=n_intervals).ivl()
    return [df_a], {}

@benchmark
def act_0(df_a):
    return df_a.ivl.basic.combine()

@benchmark
def act_1(df_a):
    return df_a.ivl.combine()

@skipif_pytest_acceptance_not_set
def test_combine():
    exps_n_intervals = [
        20,
        100,
        500,
        1000,
        2000,
        100000,
        200000,
        500000,
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
