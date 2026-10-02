from collections import Counter, defaultdict


def contains_duplicate_set(nums):
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False


def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []


def is_anagram_dict(s, t):
    if len(s) != len(t):
        return False
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    for ch in t:
        counts[ch] = counts.get(ch, 0) + 1
        if counts[ch] < 0:
            return False
    return True


def group_anagrams(strs):
    groups = defaultdict(list)
    for word in strs:
        key = "".join(sorted(word))
        groups[key].append(word)
    return list(groups.values())


def top_k_frequent_buckets(nums, k):
    counts = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, freq in counts.items():
        buckets[freq].append(num)
    result = []
    for freq in range(len(buckets) - 1, 0, -1):
        for num in buckets[freq]:
            result.append(num)
            if len(result) == k:
                return result
    return result


def intersect(nums1, nums2):
    counts = Counter(nums2)
    result = []
    for value in nums1:
        if counts[value] > 0:
            result.append(value)
            counts[value] -= 1
    return result


def product_except_self(nums):
    n = len(nums)
    result = [1] * n
    acc = 1
    for i in range(n):
        result[i] = acc
        acc = acc * nums[i]
    acc = 1
    for i in reversed(range(n)):
        result[i] = result[i] * acc
        acc = acc * nums[i]
    return result


EMPTY, SIZE, BOX = ".", 9, 3


def is_valid_sudoku_one_pass(board):
    rows = [set() for _ in range(SIZE)]
    cols = [set() for _ in range(SIZE)]
    boxes = [set() for _ in range(SIZE)]
    for r in range(SIZE):
        for c in range(SIZE):
            value = board[r][c]
            if value == EMPTY:
                continue
            box = (r // BOX) * BOX + c // BOX
            if value in rows[r] or value in cols[c] or value in boxes:
                return False
            rows[r].add(value)
            cols[c].add(value)
            boxes[box].add(value)
    return True


def is_palindrome(s):
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


def two_sum_sorted(numbers, target):
    left, right = 0, len(numbers) - 1
    while left < right:
        total = numbers[left] + numbers[right]
        if total == target:
            return [left + 1, right + 1]
        if total < target:
            left += 1
        else:
            right -= 1
    return []


def longest_substr_no_repeat(s):
    last_seen = {}
    window_start = 0
    longest = 0
    for window_end, ch in enumerate(s):
        if ch in last_seen:
            window_start = max(last_seen[ch] + 1, window_start)
        last_seen[ch] = window_end
        longest = max(longest, window_end - window_start + 1)


def valid_parentheses(s):
    pairs = {")": "(", "}": "{", "]": "["}
    stack = []
    for ch in s:
        if ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
        else:
            stack.append(ch)
    return not stack


class MinStack:
    def __init__(self):
        self.stack = []

    def push(self, val):
        smallest = val if not self.stack else min(val, self.stack[-1][1])
        self.stack.append((val, smallest))

    def pop(self):
        self.stack.pop()

    def top(self):
        return self.stack[-1][0]

    def get_min(self):
        return self.stack[-1][1]


def binary_search(nums, target):
    low, high = 0, len(nums) - 1
    while low <= high:
        mid = (low + high) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
        return -1


def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False
    rows, cols = len(matrix), len(matrix[0])
    low, high = 0, rows * cols - 1
    while low <= high:
        mid = (low + high) // 2
        value = matrix[mid // cols][mid % cols]
        if value == target:
            return True
        if value < target:
            low = mid + 1
        else:
            high = mid - 1
        return False
