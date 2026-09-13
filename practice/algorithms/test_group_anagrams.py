import pytest
from group_anagrams import group_anagrams

CASES = [
    (
        ["eat", "tea", "tan", "ate", "nat", "bat"],
        [["tan", "nat"], ["eat", "tea", "ate"], ["bat"]],
    ),
    ([], []),
    (["abc", "bca", "cab"], [["abc", "bca", "cab"]]),
    (["abc", "def", "ghi"], [["abc"], ["def"], ["ghi"]]),
    ([""], [[""]]),
    (["a"], [["a"]]),
]


def normalize_groups(groups):
    return sorted([sorted(group) for group in groups])


@pytest.mark.parametrize("strs, expected", CASES)
def test_group_anagrams(strs, expected):
    assert normalize_groups(group_anagrams(strs)) == normalize_groups(expected)
