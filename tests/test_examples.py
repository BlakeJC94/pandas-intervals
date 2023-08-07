import random

import numpy as np
import pandas as pd
import pytest

from pandas_intervals.examples import LabelsFrame, RegionsFrame


# Expected parsed object
@pytest.fixture
def labels_frame():
    return LabelsFrame(
        data=[
            [0.0, 100.0, "tag1", "study1", 0.0, "some text"],
            [0.0, 100.0, "tag2", "study1", 0.0, "some text"],
            [200.0, 350.0, "tag1", "study2", 0.0, ""],
            [1000.0, 2000.0, "undefined", "study2", 0.75, ""],
        ],
        columns=["start", "end", "tag", "study_id", "confidence", "note"],
    )


@pytest.fixture
def another_labels_frame():
    return LabelsFrame(
        data=[
            [0.0, 100.0, "tag1", "study3", 0.0, "some text"],
            [100.0, 200.0, "tag2", "study1", 0.0, "some text"],
            [100.0, 200.0, "tag1", "study1", 0.0, "other text"],
            [200.0, 350.0, "tag3", "study2", 0.0, ""],
            [1000.0, 2000.0, "undefined", "study2", 0.75, ""],
        ],
        columns=["start", "end", "tag", "study_id", "confidence", "note"],
    )


# Input data sources
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


class TestLabelsFrame:
    """Tests for the `LabelsFrame` class."""

    def test_format_labels(self, labels_df, labels_frame):
        """Test a `LabelsFrame` can be created from a `DataFrame` of labels."""
        labels = LabelsFrame(labels_df).format()
        pd.testing.assert_frame_equal(labels, labels_frame)

    # def test_labels_format_missing_required_column(self, labels_df, labels_frame):
    #     """Test an exception is raised trying to create a `LabelsFrame` with a column missing."""
    #     # Drop one of the required columns from input DataFrame
    #     drop_col = random.choice(labels_frame.required_cols)
    #     partial_labels_df = labels_df.drop(drop_col, axis=1)

    #     with pytest.raises(ValueError):
    #         LabelsFrame(partial_labels_df).format()

    def test_empty_labels_frame(self, labels_frame):
        """Test an empty `LabelsFrame` can be created with correct column types."""
        empty_labels = LabelsFrame.empty_frame()
        assert isinstance(empty_labels, LabelsFrame)
        assert len(empty_labels) == 0
        assert empty_labels.columns.tolist() == [
            *LabelsFrame.required_cols,
            *LabelsFrame.additional_cols,
        ]
        assert (empty_labels.dtypes == labels_frame.dtypes).all()

    def test_union_labels_frame(self, labels_frame, another_labels_frame):
        """Test a `LabelsFrame` can be unioned with another `LabelsFrame`."""
        result_a = labels_frame | another_labels_frame
        result_b = another_labels_frame | labels_frame
        pd.testing.assert_frame_equal(
            result_a.reset_index(drop=True),
            result_b.reset_index(drop=True),
        )
        assert (
            len(result_a) == len(labels_frame) + len(another_labels_frame) - 1
        ), "Result contains duplicate."

    def test_intersection_labels_frame(self, labels_frame, another_labels_frame):
        """Test a `LabelsFrame` can be intersectioned with another `LabelsFrame`."""
        result_a = labels_frame & another_labels_frame
        result_b = another_labels_frame & labels_frame
        pd.testing.assert_frame_equal(
            result_a.reset_index(drop=True),
            result_b.reset_index(drop=True),
        )
        assert len(result_a) == 1


# Test data
@pytest.fixture
def regions_df():
    return pd.DataFrame(
        data=[
            [0, 100, "study1", "train"],
            [200, 350.0, "study2", "val"],
            [1000, 2000.0, None, None],
        ],
        columns=["start", "end", "study_id", "category"],
    )


@pytest.fixture
def regions_frame():
    return RegionsFrame(
        data=[
            [0.0, 100.0, "study1", "train"],
            [200.0, 350.0, "study2", "val"],
            [1000.0, 2000.0, "", ""],
        ],
        columns=["start", "end", "study_id", "category"],
    )


@pytest.fixture
def another_regions_frame():
    return RegionsFrame(
        data=[
            [500.0, 600.0, "study1", "train"],
            [0.0, 100.0, "study2", "train"],
            [200.0, 350.0, "study2", "test"],
            [1000.0, 2000.0, "", ""],
        ],
        columns=["start", "end", "study_id", "category"],
    )


class TestRegionsFrame:
    """Tests for the `RegionsFrame` class."""

    def test_format_regions(self, regions_df, regions_frame):
        """Test a `RegionsFrame` can be created from a `DataFrame` of regions."""
        regions = RegionsFrame(regions_df).format()
        pd.testing.assert_frame_equal(regions, regions_frame)

    def test_regions_format_missing_required_column(self, regions_df, regions_frame):
        """Test an exception is raised trying to create a `RegionsFrame` with a column missing."""
        # Drop one or more required columns from input DataFrame
        num_cols_to_drop = random.choice(range(1, len(regions_frame.required_cols)))
        drop_cols = np.random.choice(
            regions_frame.required_cols, num_cols_to_drop, replace=False
        )
        partial_regions_df = regions_df[
            [col for col in regions_df.columns if col not in drop_cols]
        ]

        with pytest.raises(ValueError):
            LabelsFrame(partial_regions_df).format()

    def test_union_regions_frame(self, regions_frame, another_regions_frame):
        """Test a `RegionsFrame` can be unioned with another `RegionsFrame`."""
        result_a = regions_frame | another_regions_frame
        result_b = another_regions_frame | regions_frame
        pd.testing.assert_frame_equal(
            result_a.reset_index(drop=True),
            result_b.reset_index(drop=True),
        )
        assert (
            len(result_a) == len(regions_frame) + len(another_regions_frame) - 1
        ), "Result contains duplicate."

    def test_intersection_regions_frame(self, regions_frame, another_regions_frame):
        """Test a `RegionsFrame` can be intersectioned with another `RegionsFrame`."""
        result_a = regions_frame & another_regions_frame
        result_b = another_regions_frame & regions_frame
        pd.testing.assert_frame_equal(
            result_a.reset_index(drop=True),
            result_b.reset_index(drop=True),
        )
        assert len(result_a) == 1


# TODO Reimplement
# def test_raise_union_labels_frame_regions_frame(labels_frame, regions_frame):
#     with pytest.raises(ValueError):
#         _ = labels_frame | regions_frame


# TODO Reimplement
# def test_raise_intersection_labels_frame_regions_frame(labels_frame, regions_frame):
#     with pytest.raises(ValueError):
#         _ = labels_frame & regions_frame


# TODO REFACTOR
class TestCombineCloseLabels:
    @pytest.mark.parametrize(
        "labels, expected",
        [
            # Each label has a different tag - no combination
            (
                dict(start=[100, 250, 400], end=[200, 350, 500], tag=list("abc")),
                dict(start=[100, 250, 400], end=[200, 350, 500], tag=list("abc")),
            ),
            # All with same tag and within gap_size, but not sorted - all are combined
            (
                dict(start=[250, 400, 100], end=[350, 500, 200], tag=list("aaa")),
                dict(start=[100], end=[500], tag=["a"]),
            ),
            # All with same tag and two nested within but not close to each other - all are combined
            (
                dict(
                    start=[100, 250, 400, 800],
                    end=[600, 260, 410, 900],
                    tag=list("aaaa"),
                ),
                dict(start=[100, 800], end=[600, 900], tag=["a", "a"]),
            ),
            # All with same tag and two nested within but close to each other - all are combined
            (
                dict(start=[100, 250, 400], end=[600, 350, 410], tag=list("aaa")),
                dict(start=[100], end=[600], tag=["a"]),
            ),
            # Combined labels should have the maximum confidence
            (
                dict(
                    start=[100, 400],
                    end=[350, 500],
                    tag=list("aa"),
                    confidence=[0.4, 0.6],
                ),
                dict(start=[100], end=[500], tag=["a"], confidence=[0.6]),
            ),
            # Combined labels should included the joined, comma-separated notes from each label
            (
                dict(
                    start=[100, 250, 400],
                    end=[200, 350, 500],
                    tag=list("aab"),
                    note=list("zxz"),
                ),
                dict(
                    start=[100, 400], end=[350, 500], tag=list("ab"), note=["x, z", "z"]
                ),
            ),
            # Combined comma-seperated notes should also be a comma-seperated note with no repeats
            (
                dict(
                    start=[100, 250, 400],
                    end=[200, 350, 500],
                    tag=list("aab"),
                    note=["z, x", "z", "z"],
                ),
                dict(
                    start=[100, 400], end=[350, 500], tag=list("ab"), note=["x, z", "z"]
                ),
            ),
        ],
    )
    def test_label_group_combine_close(self, labels, expected):
        """Test the LabelGroup._combine_close_labels method."""
        gap_size = 50
        labels = LabelsFrame(labels).format()
        labels = labels + gap_size
        out = +labels
        out = out - gap_size
        labels = labels - gap_size
        expected_frame = LabelsFrame(expected).format()
        assert out.equals(expected_frame)
