import pytest
from anagram_palindrome import (
    is_anagram,
    is_anagram_counter,
    is_anagram_dict,
    is_palindrome,
    is_palindrome_while,
)

CASES = [
    ("anagram", "nagaram", True),
    ("rat", "car", False),
    ("listen", "silent", True),
    ("hello", "world", False),
    ("aabbcc", "abcabc", True),
    ("abcd", "dcba", True),
    ("abcde", "fghij", False),
    ("aabb", "abbb", False),
    ("", "", True),
    ("a", "a", True),
    ("a", "b", False),
    ("aab", "ab", False),
]

PALINDROME_CASES = [
    ("A man, a plan, a canal: Panama", True),
    ("race a car", False),
    ("", True),
    ("    ", True),
    (",.,.", True),
    ("a", True),
    ("ab", False),
    ("0P", False),
    ("12321", True),
]


@pytest.mark.parametrize("s, t, expected", CASES)
@pytest.mark.parametrize(
    "func",
    [is_anagram, is_anagram_dict, is_anagram_counter],
    ids=["is_anagram", "is_anagram_dict", "is_anagram_counter"],
)
def test_is_anagram(func, s, t, expected):
    assert func(s, t) == expected


@pytest.mark.parametrize("s, expected", PALINDROME_CASES)
@pytest.mark.parametrize(
    "func",
    [is_palindrome, is_palindrome_while],
    ids=["is_palindrome", "is_palindrome_while"],
)
def test_is_palindrome(func, s, expected):
    assert func(s) == expected
