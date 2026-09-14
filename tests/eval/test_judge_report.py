"""The pairing the report does before it counts anything."""

from judge_report import axis_pairs


def rows():
    # (record, refused, fabricated) - the judge disagrees with the label on
    # refused and agrees on fabricated, so an off-by-one between the two
    # columns cannot pass unnoticed.
    return [
        (
            {"labels": {"refused": True, "fabricated": False}},
            False,
            False,
        )
    ]


def test_each_axis_is_paired_with_its_own_column():
    assert axis_pairs(rows(), "refused") == [(True, False)]
    assert axis_pairs(rows(), "fabricated") == [(False, False)]


def test_an_unjudged_axis_stays_none():
    unjudged = [({"labels": {"refused": True, "fabricated": None}}, None, None)]
    assert axis_pairs(unjudged, "refused") == [(True, None)]
    assert axis_pairs(unjudged, "fabricated") == [(None, None)]
