import pytest
from merge_intervals import merge_intervals

CASES = [
    ([[1, 3], [2, 6], [8, 10], [15, 18]], [[1, 6], [8, 10], [15, 18]]),
    ([[1, 4], [4, 5]], [[1, 5]]),
    ([[1, 10], [2, 3]], [[1, 10]]),
    ([], []),
    ([[5, 7]], [[5, 7]]),
    ([[8, 10], [1, 3], [2, 6]], [[1, 6], [8, 10]]),
    ([[1, 4], [0, 4]], [[0, 4]]),
    ([[1, 4], [2, 3], [3, 9]], [[1, 9]]),
]


@pytest.mark.parametrize("intervals, merged_intervals", CASES)
def test_merge_intervals(intervals, merged_intervals):
    assert merge_intervals(intervals) == merged_intervals
