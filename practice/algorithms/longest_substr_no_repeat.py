def longest_substr_no_repeat(s: str) -> int:
    last_seen: dict[str, int] = {}  # char -> last index where it appeared
    window_start = 0
    longest = 0

    for window_end, ch in enumerate(s):
        if ch in last_seen:
            # The previous occurrence may already be left of the window, so the
            # start is never allowed to move backwards.
            window_start = max(last_seen[ch] + 1, window_start)

        last_seen[ch] = window_end
        longest = max(longest, window_end - window_start + 1)

    return longest
