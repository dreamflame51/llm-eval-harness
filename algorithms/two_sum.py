def two_sum_brute(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
            return []

def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []


# print(two_sum_brute([2, 7, 11, 15], 9))
# print(two_sum([2, 7, 11, 15], 9))
# print(two_sum([3, 3], 6))
# print(two_sum([1,2,3], 100))
#print(two_sum([2, 7, 11, 15], 18))