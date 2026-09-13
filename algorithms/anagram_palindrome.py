from collections import Counter


# O(n log n) time complexity, O(n) space complexity
def is_anagram(s, t):
    if len(s) != len(t):
        return False
    return sorted(s) == sorted(t)


# O(n) time complexity, O(min(n, k)) space complexity
def is_anagram_dict(s, t):
    if len(s) != len(t):
        return False
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    for ch in t:
        counts[ch] = counts.get(ch, 0) - 1
        if counts[ch] < 0:
            return False
    return True


# O(n) time complexity, O(min(n, k)) space complexity
def is_anagram_counter(s, t):
    return Counter(s) == Counter(t)


# O(n) time, O(n) space
def is_palindrome(s):
    cleaned = "".join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]


# O(n) time complexity, O(1) space complexity
def is_palindrome_while(s):
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():
            left = left + 1
        while left < right and not s[right].isalnum():
            right = right - 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True
