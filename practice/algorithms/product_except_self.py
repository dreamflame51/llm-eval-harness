# O(n) time (two passes), O(1) extra space (output not counted)
def product_except_self(nums: list[int]) -> list[int]:
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


def product_except_self2(nums: list[int]) -> list[int]:
    n = len(nums)
    answer = [1] * n

    prefix = 1
    for i in range(n):
        answer[i] = prefix
        prefix *= nums[i]

    suffix = 1
    for i in range(n - 1, -1, -1):
        answer[i] *= suffix
        suffix *= nums[i]

    return answer


# Same algorithm folded into a single loop: two accumulators run towards each
# other, i from the left, j from the right. Same O(n), just denser — and harder
# to read, which is why the two-pass version above is the main one.
#
# def product_except_self_one_loop(nums: list[int]) -> list[int]:
#     n = len(nums)
#     result = [1] * n
#     left = right = 1
#     for i in range(n):
#         result[i] = result[i] * left
#         left = left * nums[i]
#
#         j = n - 1 - i
#         result[j] = result[j] * right
#         right = right * nums[j]
#
#     return result
