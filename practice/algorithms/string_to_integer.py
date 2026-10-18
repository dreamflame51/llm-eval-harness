INT_MIN = -(2**31)
INT_MAX = 2**31 - 1


# Time: O(n) / # Space: O(1)
def my_atoi(s: str) -> int:
    """LeetCode 8. Four phases in a fixed order: spaces, sign, digits, clamp.

    The order is the whole problem: a phase may only read what the previous
    phases left behind, and none of them may run twice.
    """
    i = 0
    n = len(s)

    # 1. spaces: only ' ' counts as whitespace here, and only before everything else
    while i < n and s[i] == " ":
        i += 1

    # 2. sign: at most one, and only in this position. What is sign when there is none?
    sign = 1
    # TODO: if the current character is '+' or '-', record it and step over it
    if i < n and s[i] in "+-":
        sign = -1 if s[i] == "-" else 1
        i += 1

    # 3. digits: keep reading while the character is one.
    #    Use "0" <= ch <= "9", not ch.isdigit() -- the latter is True for '²' too.
    #    A new digit turns 4 into 42: what arithmetic is that?
    value = 0
    while i < n and "0" <= s[i] <= "9":
        value = value * 10 + int(s[i])
        i += 1

    # 4. clamp: the result never leaves [INT_MIN, INT_MAX].
    #    Both ends are reachable, and they are not symmetric.
    result = sign * value
    if result < INT_MIN:
        return INT_MIN
    if result > INT_MAX:
        return INT_MAX
    return result


# Time: ?
# Space: ?
