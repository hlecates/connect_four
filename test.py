from utils import EMPTY, COLUMNS, ROWS, HUMAN, AI, detect_win, is_terminal_board, encode_state


def decode_state(state):
        """
        Given a state (likely a tuple), decode it into a board matrix if needed.
        If encode_state is consistent, we can directly reshape the state into a board.
        """
        # encode_state returns a tuple of length ROWS*COLUMNS, representing board row-major order.
        # We can convert it back into a 2D list.
        board_list = list(state)
        # Rebuild into a board
        board = [board_list[i*COLUMNS:(i+1)*COLUMNS] for i in range(ROWS)]
        return board
'''
test_board = [[EMPTY for _ in range(COLUMNS)] for _ in range(ROWS)]
# Assume AI = 2
test_board[5][0] = 2
test_board[5][1] = 2
test_board[5][2] = 2
test_board[5][3] = 2
print(is_terminal_board(test_board))
print("AI wins horizontally:", detect_win(test_board, 2))  # Should print True

test_board = [[0]*7 for _ in range(6)]  # An empty 6x7 board if 0 is EMPTY
print(detect_win(test_board, HUMAN))  # Should be False
print(detect_win(test_board, AI))     # Should be False
'''
test_board = [[0]*7 for _ in range(6)]
test_board[5][3] = 2
encoded = encode_state(test_board)
decoded = decode_state(encoded)
assert decoded == test_board, "decode_state does not return the original board"