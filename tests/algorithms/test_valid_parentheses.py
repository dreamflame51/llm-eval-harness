import pytest

from valid_parentheses import valid_parentheses

cases = [
    ("()", True),
    ("()[]{}", True),
    ("(]", False),
    ("([)]", False),
    ("{[]}", True),
    ("", True),
    ("(", False),
    (")", False),
    ("{[()]}", True),
    ("{[(])}", False),
]


@pytest.mark.parametrize("s,expected", cases)
def test_valid_parentheses(s, expected):
    assert valid_parentheses(s) == expected
