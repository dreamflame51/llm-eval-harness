# O(n log n) time from the sort, O(n) space
def merge_intervals(interval_list):
    sorted_intervals = sorted(interval_list, key=lambda x: x[0])
    result = []
    for start, end in sorted_intervals:
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return result
