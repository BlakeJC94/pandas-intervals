import pytest

import pandas as pd

from pandas_intervals import IntervalsFrame

#        0         0         0         0         0         0         0
#     A:     (----]    (----]         (--------------]
#     B: (--]   (---]          (----]    (---]    (------] (---]
#
# A ∩ B:     (----]    (----]         (--------------]
#
#        0         0         0         0         0         0         0
#     A:     (----]    (----]         (--------------]
#     B: (--]   (---]          (----]    (---]    (------] (---]
#
# A ∪ B:     (----]    (----]         (--------------]
#        (--]   (---]          (----]    (---]    (------] (---]
#
#        0         0         0         0         0         0         0
#     A:     (----]    (----]         (--------------]
#     B: (--]   (---]          (----]    (---]    (------] (---]
#
# A + B: (--] (-----]   (----] (----]  (-----------------] (---]
#
#        0         0         0         0         0         0         0
#     A:     (----]    (----]         (--------------]
#     B: (--]   (---]          (----]    (---]    (------] (---]
#
# A - B:     (--]      (----]         (--]   (----]

@pytest.fixture
def intervals_frame_a():
    return IntervalsFrame(
        [
            [50, 100],
            [150, 200],
            [300, 450],
        ],
    ).format()

@pytest.fixture
def intervals_frame_b():
    df = pd.DataFrame(
        [
            [10, 40],
            [80, 120],
            [230, 280],
            [330, 370],
            [420, 490],
            [510, 550],
        ],
        columns=["start", "end"],
    )
    return IntervalsFrame(df).format()

class TestIntervalsFrame:
    def test_init_types_and_columns(self, intervals_frame_a, intervals_frame_b):
        expected_columns = ["start", "end"]
        for vf in [intervals_frame_a, intervals_frame_b]:
            assert vf.columns.tolist() == expected_columns
            assert vf.dtypes[expected_columns].tolist() ==  [float, float]

    # def test_union(self, intervals_frame_a, intervals_frame_b):
    #     ...
