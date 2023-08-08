import pandas as pd

import pytest

@pytest.fixture
def labels_df():
    return pd.DataFrame(
        data=[
            [0, 100, "study1", "tag1", 0.0, "some text"],
            [0, 100, "study1", "tag2", 0.0, "some text"],
            [200, 350, "study2", "tag1", 0.0, ""],
            [1000, 2000, "study2", "undefined", 0.75, ""],
        ],
        columns=["start", "end", "study_id", "tag", "confidence", "note"],
    )

def test_intervals_accessor(labels_df):
    labels_df.ivl()
    ...
