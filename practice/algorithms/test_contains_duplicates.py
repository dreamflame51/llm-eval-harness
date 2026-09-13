import pytest
from contains_duplicate import (
    contains_duplicate_brute_force,
    contains_duplicate_set,
    contains_duplicate_sorted,
)


@pytest.mark.parametrize(
    "nums, expected",
    [
        ([1, 2, 3, 1], True),
        ([1, 2, 3, 4], False),
        ([1, 1], True),
        ([], False),
        ([5], False),
    ],
    ids=[
        "has_duplicates",
        "no_duplicates",
        "two_duplicates",
        "empty_list",
        "single_element",
    ],
)
@pytest.mark.parametrize(
    "func",
    [contains_duplicate_brute_force, contains_duplicate_sorted, contains_duplicate_set],
    ids=["brute_force", "sorted", "set"],
)
def test_contains_duplicate(nums, func, expected):
    assert func(nums) == expected


def test_sorted_version_does_not_reorder_caller_list():
    nums = [3, 1, 2]
    contains_duplicate_sorted(nums)
    assert nums == [3, 1, 2]
