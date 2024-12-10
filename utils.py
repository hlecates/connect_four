import numpy as np
import random
import sys
import inspect

ROWS = 6
COLUMNS = 7

EMPTY = 0
HUMAN = 1
AI = 2

MAX_SPACE_TO_WIN = 3

def create_board():
    return np.zeros((ROWS, COLUMNS), dtype=np.int8)

def is_valid_column(board, column):
    return board[0][column] == EMPTY

def valid_locations(board):
    return [c for c in range(COLUMNS) if is_valid_column(board, c)]

def place_piece(board, player, column):
    for row in reversed(range(ROWS)):
        if board[row][column] == EMPTY:
            board[row][column] = player
            return

def clone_and_place_piece(board, player, column):
    new_board = board.copy()
    place_piece(new_board, player, column)
    return new_board

def detect_win(board, player):
    # Horizontal win
    for col in range(COLUMNS - MAX_SPACE_TO_WIN):
        for row in range(ROWS):
            if board[row][col] == player and board[row][col+1] == player and \
                    board[row][col+2] == player and board[row][col+3] == player:
                return True
    # Vertical win
    for col in range(COLUMNS):
        for row in range(ROWS - MAX_SPACE_TO_WIN):
            if board[row][col] == player and board[row+1][col] == player and \
                    board[row+2][col] == player and board[row+3][col] == player:
                return True
    # Diagonal upwards win
    for col in range(COLUMNS - MAX_SPACE_TO_WIN):
        for row in range(ROWS - MAX_SPACE_TO_WIN):
            if board[row][col] == player and board[row+1][col+1] == player and \
                    board[row+2][col+2] == player and board[row+3][col+3] == player:
                return True
    # Diagonal downwards win
    for col in range(COLUMNS - MAX_SPACE_TO_WIN):
        for row in range(MAX_SPACE_TO_WIN, ROWS):
            if board[row][col] == player and board[row-1][col+1] == player and \
                    board[row-2][col+2] == player and board[row-3][col+3] == player:
                return True
    return False

def is_terminal_board(board):
    return detect_win(board, HUMAN) or detect_win(board, AI) or len(valid_locations(board)) == 0

def encode_state(board):
    board_array = np.array(board)
    return tuple(board_array.flatten())

class Counter(dict):
    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        for key in keys:
            self[key] += count

    def argMax(self):
        if len(self.keys()) == 0:
            return None
        all_items = list(self.items())
        values = [x[1] for x in all_items]
        maxIndex = values.index(max(values))
        return all_items[maxIndex][0]

    def totalCount(self):
        return sum(self.values())

    def normalize(self):
        total = float(self.totalCount())
        if total == 0:
            return
        for key in list(self.keys()):
            self[key] = self[key] / total

    def divideAll(self, divisor):
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        return Counter(dict.copy(self))

def flipCoin(p):
    r = random.random()
    return r < p

def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]
    print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)


def count_threats(board, player):
    threats = 0
    EMPTY = 0

    # Check horizontal threats
    for col in range(COLUMNS - 3):
        for row in range(ROWS):
            window = board[row, col:col+4]
            if (list(window).count(player) == 3 and list(window).count(EMPTY) == 1):
                threats += 1

    # Check vertical threats
    for col in range(COLUMNS):
        for row in range(ROWS - 3):
            window = board[row:row+4, col]
            if (list(window).count(player) == 3 and list(window).count(EMPTY) == 1):
                threats += 1

    # Check positively sloped diagonals
    for col in range(COLUMNS - 3):
        for row in range(ROWS - 3):
            window = [board[row+i][col+i] for i in range(4)]
            if window.count(player) == 3 and window.count(EMPTY) == 1:
                threats += 1

    # Check negatively sloped diagonals
    for col in range(COLUMNS - 3):
        for row in range(3, ROWS):
            window = [board[row - i][col + i] for i in range(4)]
            if window.count(player) == 3 and window.count(EMPTY) == 1:
                threats += 1

    return threats
