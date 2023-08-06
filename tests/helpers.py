from __future__ import annotations
from typing import Union, List, Mapping, Optional

import pandas as pd


def labels_from_str(
    labels_str: Union[str, List[str]],
    tags_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Create an labels DataFrame from a string input.

    Label starts are represented as '(' and ends are represented as ')' characters. '|' characters
    represent a label start and an end, therefore can be used to represent a zero-duration label or
    start a new label that starts at the previous label end. An alphabetic character after a label
    start is interpreted as the tag (if not given, a default tag '-' is inserted).

    Limitations:
        - Zero-length labels can't have tags added via string when adjacent to the next label,
        - Overlapping labels aren't generally supported, use `pd.concat` across multiple outputs,
        - 2-character labels can't be tagged.

    Examples:
        - Simple labels (some with tags):
            " (---)  (q----)    (w--) (------)  (--) "
        - Zero duration labels:
            "    (q----)   |w   |  |     |e   (----) "
        - Tangential labels:
            "  (q----|-----|w----|e----)     (----)  "
        - Simple, tangential, and zero duration labels:
            "     (q---)   (w--|e----)   |  |        "

    Args:
        labels_str: Labels in string format. Can also accept a list of labels in string format,
            which can be used for clearer construction label DataFrames of overlapping labels.
        tags_map: An optional mapping from single character tags to longer strings

    Returns:
        DataFrame containing columns `'start', 'end', 'tag'`.
    """
    if isinstance(labels_str, list):
        return pd.concat([labels_from_str(s) for s in labels_str])
    if not isinstance(labels_str, str):
        raise ValueError("Input must be a string")

    starts, ends, tags = [], [], []
    default_tag = "-"
    i, start = 0, None
    for i, c in enumerate(labels_str):
        next_i = min(i + 1, len(labels_str) - 1)
        if c == "(":  # Start new label
            start = i * 100
            label_tags = [default_tag] if not labels_str[next_i].isalnum() else []
        if c == ")":  # End a label
            end = i * 100
            _end_label(starts, ends, tags, start, end, label_tags)
            start = None
        if (
            c == "|"
        ):  # Either a zero-duration label, or start a tangential label + end the previous
            if start is not None:
                _end_label(starts, ends, tags, start, i * 100, label_tags)
            start = end = i * 100
            label_tags = [default_tag] if not labels_str[next_i].isalnum() else []
        elif c.isalnum():  # A tag to add to the label
            label_tags.append(c)
        elif c == " " and start is not None:  # End a zero-duration label
            _end_label(starts, ends, tags, start, end, label_tags)
            start = None
    if start is not None:
        _end_label(starts, ends, tags, start, i * 100, label_tags)

    labels = pd.DataFrame(
        list(zip(starts, ends, tags)), columns=["start", "end", "tag"]
    )
    labels[["start", "end"]] = labels[["start", "end"]].astype(float)
    if tags_map:
        labels["tag"] = labels["tag"].replace(tags_map)
    return labels


def _end_label(starts, ends, tags, start, end, label_tags):
    for tag in label_tags:
        starts.append(start)
        ends.append(end)
        tags.append(tag)
