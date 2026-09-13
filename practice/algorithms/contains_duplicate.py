def contains_duplicate_brute_force(nums):
    # Time: O(n^2), Space: O(1)
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                return True
    return False


# Time: O(n log n), Space: O(n)
def contains_duplicate_sorted(nums):
    sorted_nums = sorted(nums)
    for i in range(len(sorted_nums) - 1):
        if sorted_nums[i] == sorted_nums[i + 1]:
            return True
    return False


# Time: O(n), Space: O(n)
def contains_duplicate_set(nums):
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False
