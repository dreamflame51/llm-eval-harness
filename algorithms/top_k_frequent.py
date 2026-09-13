from collections import Counter


# O(n log n) time, O(n) space
def top_k_frequent(nums, k):
    #   return [pair[0] for pair in Counter(nums).most_common(k)]
    return [num for num, _ in Counter(nums).most_common(k)]


# O
def top_k_frequent_sorted(nums, k):
    counts = Counter(nums)
    return sorted(counts, key=lambda num: counts[num], reverse=True)[:k]


# O(n) time, O(n) space
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
