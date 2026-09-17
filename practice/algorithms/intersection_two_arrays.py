from collections import Counter


# Time: O(n + m), Space: O(m)
def intersect(nums1: list[int], nums2: list[int]) -> list[int]:
    counts = Counter(nums2)
    result = []

    for value in nums1:
        # 1. is this value still available in counts?
        if counts[value] > 0:
            # 2. take it into the answer
            result.append(value)
            # 3. spend one unit of it
            counts[value] -= 1

    return result
