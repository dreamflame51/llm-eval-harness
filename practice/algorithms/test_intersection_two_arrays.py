import pytest
from intersection_two_arrays import intersect

cases = [
    ([1, 2, 2, 1], [2, 2], [2, 2]),
    ([4, 9, 5], [9, 4, 9, 8, 4], [4, 9]),
    ([1, 1], [1], [1]),
    ([1, 2, 3], [4, 5, 6], []),
    ([1], [1], [1]),
]


@pytest.mark.parametrize("nums1, nums2, expected", cases)
def test_intersect(nums1, nums2, expected):
    assert sorted(intersect(nums1, nums2)) == sorted(expected)
