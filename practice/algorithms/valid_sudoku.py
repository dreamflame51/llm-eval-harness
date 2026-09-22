EMPTY = "."
SIZE = 9
BOX = 3


def has_duplicate(cells: list[str]) -> bool:
    """One group of nine cells: True if a digit repeats. Dots are ignored."""
    seen = set()
    for value in cells:
        # 1. an empty cell is never a duplicate -> move on to the next one
        if value in seen:
            return True
        if value == EMPTY:
            continue

        # 2. from here it is contains_duplicate_set, unchanged
        seen.add(value)

    return False


def is_valid_sudoku(board: list[list[str]]) -> bool:
    # rows: board[r] is already the list of nine cells
    for r in range(SIZE):
        if has_duplicate(board[r]):
            return False

    # columns: your turn, same shape as above
    for c in range(SIZE):
        column = [board[r][c] for r in range(SIZE)]
        if has_duplicate(column):
            return False

    # boxes: (br, bc) addresses the 3x3 grid of boxes, (i, j) the cell inside one
    for br in range(BOX):
        for bc in range(BOX):
            cells = [
                board[br * BOX + i][bc * BOX + j]
                for i in range(BOX)
                for j in range(BOX)
            ]
            if has_duplicate(cells):
                return False

    return True


# Time: O(n^2) over an n x n board = 81 cells, Space: O(n^2) for the journals
def is_valid_sudoku_one_pass(board: list[list[str]]) -> bool:
    """Same rules, one visit per cell: every cell knows all three of its groups."""
    rows = [set() for _ in range(SIZE)]
    cols = [set() for _ in range(SIZE)]
    boxes = [set() for _ in range(SIZE)]

    for r in range(SIZE):
        for c in range(SIZE):
            value = board[r][c]
            if value == EMPTY:
                continue

            box = (r // BOX) * BOX + c // BOX
            if value in rows[r] or value in cols[c] or value in boxes[box]:
                return False

            rows[r].add(value)
            cols[c].add(value)
            boxes[box].add(value)

    return True
