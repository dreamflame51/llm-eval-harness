import pytest
from valid_sudoku import has_duplicate, is_valid_sudoku, is_valid_sudoku_one_pass

VALID_BOARD = [
    "53..7....",
    "6..195...",
    ".98....6.",
    "8...6...3",
    "4..8.3..1",
    "7...2...6",
    ".6....28.",
    "...419..5",
    "....8..79",
]

IMPLEMENTATIONS = [is_valid_sudoku, is_valid_sudoku_one_pass]


def board_from(rows: list[str]) -> list[list[str]]:
    return [list(row) for row in rows]


def board_with(row_index: int, row: str) -> list[list[str]]:
    rows = list(VALID_BOARD)
    rows[row_index] = row
    return board_from(rows)


GROUP_CASES = [
    (list("53..7...8"), False),
    (list("53..7...5"), True),
    (list("........."), False),
    (list("123456789"), False),
    (list("1.......1"), True),
]

BOARD_CASES = [
    # a duplicate is invisible to two of the three rules, so each rule needs its own board
    ("valid", board_from(VALID_BOARD), True),
    ("empty", board_from(["........."] * 9), True),
    ("duplicate in a row", board_with(0, "53..7...5"), False),
    ("duplicate in a column", board_with(8, "5...8..79"), False),
    ("duplicate in a box only", board_with(1, "68.195..."), False),
]


@pytest.mark.parametrize("cells, expected", GROUP_CASES)
def test_has_duplicate(cells, expected):
    assert has_duplicate(cells) == expected


@pytest.mark.parametrize("is_valid", IMPLEMENTATIONS)
@pytest.mark.parametrize("name, board, expected", BOARD_CASES)
def test_is_valid_sudoku(name, board, expected, is_valid):
    assert is_valid(board) == expected


@pytest.mark.parametrize("is_valid", IMPLEMENTATIONS)
def test_board_is_not_modified(is_valid):
    board = board_from(VALID_BOARD)
    is_valid(board)
    assert board == board_from(VALID_BOARD)
