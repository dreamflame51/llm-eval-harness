# O(n) time (two passes), O(1) extra space (output not counted)
def product_except_self(nums: list[int]) -> list[int]:
    n = len(nums)
    result = [1] * n

    # Pass 1, left to right: acc holds the product of everything LEFT of i.
    # Write first, multiply after -> nums[i] is not in acc yet.
    acc = 1
    for i in range(n):
        result[i] = acc
        acc = acc * nums[i]

    # Pass 2, right to left: acc holds the product of everything RIGHT of i,
    # multiplied into what pass 1 already stored.
    acc = 1
    for i in range(n - 1, -1, -1):
        result[i] = result[i] * acc
        acc = acc * nums[i]

    return result


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
