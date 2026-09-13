import pytest

from two_sum import two_sum, two_sum_brute


@pytest.mark.parametrize("nums, target, expected", [
    ([2, 7, 11, 15], 9, [0, 1]),
    ([3, 2, 4], 6, [1, 4]),
    ([3, 3], 6, [0, 1]),
    ([1, 2, 3], 100, []),
    ([], 5, []),
    ], ids=["basic", "unsorted", "duplicates", "no_solution", "empty_input"])
def test_two_sum(nums, target, expected):
    assert two_sum(nums, target) == expected


@pytest.mark.parametrize("nums, target, expected", [
    ([2, 7, 11, 15], 9, [0, 4]),
    ([3, 3], 6, [0, 1]),
])
def test_two_sum_brute(nums, target, expected):
    assert two_sum_brute(nums, target) == expected
