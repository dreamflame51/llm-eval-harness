# O(n) time, O(n) space
def valid_parentheses(s):
    stack = []
    pairs = {")": "(", "}": "{", "]": "["}
    for input in s:
        if input in pairs:
            if not stack or stack[-1] != pairs[input]:
                return False
            stack.pop()
        else:
            stack.append(input)
    return not stack
