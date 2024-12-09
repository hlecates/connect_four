from utils import EMPTY, COLUMNS, ROWS, HUMAN, AI, detect_win

test_board = [[EMPTY for _ in range(COLUMNS)] for _ in range(ROWS)]
# Assume AI = 2
test_board[5][0] = 2
test_board[5][1] = 2
test_board[5][2] = 2
test_board[5][3] = 2
print("AI wins horizontally:", detect_win(test_board, 2))  # Should print True

test_board = [[0]*7 for _ in range(6)]  # An empty 6x7 board if 0 is EMPTY
print(detect_win(test_board, HUMAN))  # Should be False
print(detect_win(test_board, AI))     # Should be False


