import random
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pandas_intervals.examples
from pandas_intervals.intervals_accessor import _apply_operation_to_groups
from tests.helpers import random_intervals


# Test data
@pytest.fixture
def regions_df():
    return pd.DataFrame(
        data=[
            [0.0, 100.0, 1, "train"],
            [200.0, 350.0, 2, "val"],
            [1000.0, 2000.0, 0, ""],
        ],
        columns=["start", "end", "tag", "note"],
    )


@pytest.fixture
def regions_json_data():
    return [
        dict(start=500.0, end=600.0, tag=1, note="train"),
        dict(start=0.0, end=100.0, tag=2, note="train"),
        dict(start=200.0, end=350.0, tag=2, note="test"),
        dict(start=1000.0, end=2000.0, tag=0, note=None),
    ]


@pytest.fixture
def regions_json_filepath(tmp_path, regions_json_data):
    filepath = Path(tmp_path) / "data.json"
    with open(filepath, "w") as f:
        json.dump(regions_json_data, f)
    yield filepath
    filepath.unlink()


random_regions = partial(
    random_intervals,
    random_fields=[("tag", [0, 1, 2]), ("note", [None, "train", "val"])],
)


class TestRegionsAccessor:
    def test_constructor(self, regions_json_filepath, regions_json_data):
        filepath = random.choice(
            [Path(regions_json_filepath), str(regions_json_filepath)]
        )
        expected = pd.DataFrame(regions_json_data).reg()

        result = pd.DataFrame.reg.from_json(filepath)

        pd.testing.assert_frame_equal(result, expected)

    def test_raise_format_missing_added_required_column(self, regions_df):
        drop_col = "tag"  # Not in pd.DataFrame.reg.default_values.keys
        partial_intervals_df = regions_df.drop(drop_col, axis=1)
        with pytest.raises(ValueError):
            partial_intervals_df.reg()

    def test_default_value_parsed(self, regions_df):
        drop_col = "note"  # In pd.DataFrame.reg.default_values.keys
        regions_df = regions_df.drop(drop_col, axis=1)

        regions_df = regions_df.reg()

        assert (regions_df["note"] == pd.DataFrame.reg.default_values["note"]).all()

    def test_empty(self):
        result = random.choice([pd.DataFrame().reg(), pd.DataFrame.reg.empty()])
        assert len(result) == 0
        assert result.columns.tolist()[:3] == pd.DataFrame.reg.required_cols
        assert all(
            result.dtypes[col] == dtype for col, dtype, _ in pd.DataFrame.reg.fields
        )

    def test_intervals_contains(self):
        df_a = random_regions(
            n_intervals=random.randint(0, 12),
        )

        n_selected = random.randint(0, len(df_a) // 2)
        has_other_interval = random.random() < 0.5

        random_mask = np.zeros(len(df_a), dtype=bool)
        random_mask[:n_selected] = True
        np.random.shuffle(random_mask)
        df_b = df_a[random_mask]

        if has_other_interval:
            df_b = pd.concat([df_b, random_regions(n_intervals=1)], axis=0)

        result = df_a.reg.contains(df_b)

        assert result is not has_other_interval


# class TestRegionsFrame:
#     """Tests for the `RegionsFrame` class."""

#     def test_format_regions(self, regions_df, regions_frame):
#         """Test a `RegionsFrame` can be created from a `DataFrame` of regions."""
#         regions = RegionsFrame(regions_df).format()
#         pd.testing.assert_frame_equal(regions, regions_frame)

#     def test_regions_format_missing_required_column(self, regions_df, regions_frame):
#         """Test an exception is raised trying to create a `RegionsFrame` with a column missing."""
#         # Drop one or more required columns from input DataFrame
#         num_cols_to_drop = random.choice(range(1, len(regions_frame.required_cols)))
#         drop_cols = np.random.choice(
#             regions_frame.required_cols, num_cols_to_drop, replace=False
#         )
#         partial_regions_df = regions_df[
#             [col for col in regions_df.columns if col not in drop_cols]
#         ]

#         with pytest.raises(ValueError):
#             LabelsFrame(partial_regions_df).format()

#     def test_union_regions_frame(self, regions_frame, another_regions_frame):
#         """Test a `RegionsFrame` can be unioned with another `RegionsFrame`."""
#         result_a = regions_frame | another_regions_frame
#         result_b = another_regions_frame | regions_frame
#         pd.testing.assert_frame_equal(
#             result_a.reset_index(drop=True),
#             result_b.reset_index(drop=True),
#         )
#         assert (
#             len(result_a) == len(regions_frame) + len(another_regions_frame) - 1
#         ), "Result contains duplicate."

#     def test_intersection_regions_frame(self, regions_frame, another_regions_frame):
#         """Test a `RegionsFrame` can be intersectioned with another `RegionsFrame`."""
#         result_a = regions_frame & another_regions_frame
#         result_b = another_regions_frame & regions_frame
#         pd.testing.assert_frame_equal(
#             result_a.reset_index(drop=True),
#             result_b.reset_index(drop=True),
#         )
#         assert len(result_a) == 1


# # TODO REFACTOR
# class TestCombineCloseLabels:
#     @pytest.mark.parametrize(
#         "labels, expected",
#         [
#             # Each label has a different tag - no combination
#             (
#                 dict(start=[100, 250, 400], end=[200, 350, 500], tag=list("abc")),
#                 dict(start=[100, 250, 400], end=[200, 350, 500], tag=list("abc")),
#             ),
#             # All with same tag and within gap_size, but not sorted - all are combined
#             (
#                 dict(start=[250, 400, 100], end=[350, 500, 200], tag=list("aaa")),
#                 dict(start=[100], end=[500], tag=["a"]),
#             ),
#             # All with same tag and two nested within but not close to each other - all are combined
#             (
#                 dict(
#                     start=[100, 250, 400, 800],
#                     end=[600, 260, 410, 900],
#                     tag=list("aaaa"),
#                 ),
#                 dict(start=[100, 800], end=[600, 900], tag=["a", "a"]),
#             ),
#             # All with same tag and two nested within but close to each other - all are combined
#             (
#                 dict(start=[100, 250, 400], end=[600, 350, 410], tag=list("aaa")),
#                 dict(start=[100], end=[600], tag=["a"]),
#             ),
#             # Combined labels should have the maximum confidence
#             (
#                 dict(
#                     start=[100, 400],
#                     end=[350, 500],
#                     tag=list("aa"),
#                     confidence=[0.4, 0.6],
#                 ),
#                 dict(start=[100], end=[500], tag=["a"], confidence=[0.6]),
#             ),
#             # Combined labels should included the joined, comma-separated notes from each label
#             (
#                 dict(
#                     start=[100, 250, 400],
#                     end=[200, 350, 500],
#                     tag=list("aab"),
#                     note=list("zxz"),
#                 ),
#                 dict(
#                     start=[100, 400], end=[350, 500], tag=list("ab"), note=["x, z", "z"]
#                 ),
#             ),
#             # Combined comma-seperated notes should also be a comma-seperated note with no repeats
#             (
#                 dict(
#                     start=[100, 250, 400],
#                     end=[200, 350, 500],
#                     tag=list("aab"),
#                     note=["z, x", "z", "z"],
#                 ),
#                 dict(
#                     start=[100, 400], end=[350, 500], tag=list("ab"), note=["x, z", "z"]
#                 ),
#             ),
#         ],
#     )
#     def test_label_group_combine_close(self, labels, expected):
#         """Test the LabelGroup._combine_close_labels method."""
#         gap_size = 50
#         labels = LabelsFrame(labels).format()
#         labels = labels + gap_size
#         out = +labels
#         out = out - gap_size
#         labels = labels - gap_size
#         expected_frame = Labels Frame(expected).format()
#         assert out.equals(expected_frame)
