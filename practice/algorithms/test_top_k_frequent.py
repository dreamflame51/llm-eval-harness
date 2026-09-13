import pytest
from top_k_frequent import top_k_frequent, top_k_frequent_buckets, top_k_frequent_sorted

CASES = [
    ([1, 1, 1, 2, 2, 3], 2, [1, 2]),
    ([1], 1, [1]),
    ([1, 2, 3], 1, [1]),
    ([1, 2, 3], 3, [1, 2, 3]),
    ([5, 5, 5], 1, [5]),
]


@pytest.mark.parametrize("nums, k, expected", CASES)
@pytest.mark.parametrize(
    "func",
    [top_k_frequent, top_k_frequent_sorted, top_k_frequent_buckets],
    ids=["top_k_frequent", "top_k_frequent_sorted", "top_k_frequent_buckets"],
)
def test_top_k_frequent(nums, k, expected, func):
    result = func(nums, k)
    assert sorted(result) == sorted(expected)
