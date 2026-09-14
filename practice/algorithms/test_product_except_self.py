import pytest
from product_except_self import product_except_self

CASES = [
    ([1, 2, 3, 4], [24, 12, 8, 6]),
    ([2, 3], [3, 2]),
    ([1, 0, 3], [0, 3, 0]),
    ([0, 2, 0], [0, 0, 0]),
    ([-1, 2, -2], [-4, 2, -2]),
    ([5], [1]),
    ([], []),
]
IDS = [
    "basic",
    "two_elements",
    "single_zero",
    "two_zeros",
    "negatives",
    "single_element",
    "empty",
]


@pytest.mark.parametrize("nums, expected", CASES, ids=IDS)
def test_product_except_self(nums, expected):
    assert product_except_self(nums) == expected
