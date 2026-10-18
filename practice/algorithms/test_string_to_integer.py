import pytest
from string_to_integer import INT_MAX, INT_MIN, my_atoi

CASES = [
    # from the statement
    ("42", 42),
    (" -042", -42),
    ("1337c0d3", 1337),
    ("0-1", 0),
    ("words and 987", 0),
    ("-91283472332", INT_MIN),
    # nothing to read
    ("", 0),
    ("   ", 0),
    ("+", 0),
    ("-", 0),
    (".", 0),
    ("abc", 0),
    # the sign is allowed once, in one place only
    ("+-12", 0),
    ("-+12", 0),
    ("++1", 0),
    (" 1 2", 1),
    ("+ 413", 0),
    # digits stop at the first character that is not one
    ("3.14159", 3),
    ("12a34", 12),
    ("0000000000012345678", 12345678),
    ("  -0012a42", -12),
    ("-0", 0),
    # the clamp, from both sides
    ("2147483647", INT_MAX),
    ("2147483648", INT_MAX),
    ("-2147483648", INT_MIN),
    ("-2147483649", INT_MIN),
    ("9223372036854775808", INT_MAX),
    ("-99999999999999999999", INT_MIN),
]


@pytest.mark.parametrize("s, expected", CASES)
def test_my_atoi(s, expected):
    assert my_atoi(s) == expected
