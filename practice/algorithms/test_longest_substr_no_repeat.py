import pytest
from longest_substr_no_repeat import longest_substr_no_repeat

CASES = [
    ("abcabcbb", 3),
    ("bbbbb", 1),
    ("pwwkew", 3),
    ("abba", 2),
    ("dvdf", 3),
    ("", 0),
    (" ", 1),
    ("au", 2),
    ("tmmzuxt", 5),
]


@pytest.mark.parametrize("s, expcted", CASES)
def test_longest_substr_no_repeat(s, expected):
    assert longest_substr_no_repeat(s) == expected
