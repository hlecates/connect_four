import math
import random
from utils import valid_locations, clone_and_place_piece, detect_win, is_terminal_board, AI, HUMAN, COLUMNS, ROWS, EMPTY

class MinimaxAgent:
    def __init__(self, depth=5):
        self.depth = depth

    def getAction(self, board):
        # Since this is the AI's turn and AI is the maximizing player, we start with maxValue
        col, _ = self.maxValue(board, self.depth, -math.inf, math.inf)
        return col

    def maxValue(self, board, ply, alpha, beta):
        """
        maxValue tries to maximize the AI's outcome.
        """
        terminal = is_terminal_board(board)
        if ply == 0 or terminal:
            return self.terminal_evaluation(board)

        valid_cols = valid_locations(board)
        if not valid_cols:
            # No moves left, treat as terminal draw
            return None, 0

        value = -math.inf
        col_choice = random.choice(valid_cols)
        for c in valid_cols:
            next_board = clone_and_place_piece(board, AI, c)
            _, new_score = self.minValue(next_board, ply-1, alpha, beta)
            if new_score > value:
                value = new_score
                col_choice = c
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return col_choice, value

    def minValue(self, board, ply, alpha, beta):
        """
        minValue tries to minimize the outcome from the AI's perspective, 
        i.e., it simulates the human player's best response.
        """
        terminal = is_terminal_board(board)
        if ply == 0 or terminal:
            return self.terminal_evaluation(board)

        valid_cols = valid_locations(board)
        if not valid_cols:
            # No moves left, treat as terminal draw
            return None, 0

        value = math.inf
        col_choice = random.choice(valid_cols)
        for c in valid_cols:
            next_board = clone_and_place_piece(board, HUMAN, c)
            _, new_score = self.maxValue(next_board, ply-1, alpha, beta)
            if new_score < value:
                value = new_score
                col_choice = c
            beta = min(beta, value)
            if beta <= alpha:
                break
        return col_choice, value

    def terminal_evaluation(self, board):
        """
        Evaluates the board if it's terminal or if depth limit is reached.
        Returns (column, value).
        """
        terminal = is_terminal_board(board)
        if terminal:
            if detect_win(board, HUMAN):
                return (None, -1000000000)
            elif detect_win(board, AI):
                return (None, 1000000000)
            else:
                # Draw
                return (None, 0)
        else:
            # Not terminal but depth reached
            return (None, self.evaluate_board(board))

    def evaluate_board(self, board):
        # Score the board from the AI's perspective
        return self.score(board, AI)

    def score(self, board, player):
        score = 0
        center_col = COLUMNS // 2

        # Slight preference for center column
        for row in range(ROWS):
            if board[row][center_col] == player:
                score += 3

        # Check horizontal
        for col in range(COLUMNS - 3):
            for row in range(ROWS):
                adjacent_pieces = [board[row][col+i] for i in range(4)]
                score += self.evaluate_adjacents(adjacent_pieces, player)

        # Check vertical
        for col in range(COLUMNS):
            for row in range(ROWS - 3):
                adjacent_pieces = [board[row+i][col] for i in range(4)]
                score += self.evaluate_adjacents(adjacent_pieces, player)

        # Check positively sloped diagonals
        for col in range(COLUMNS - 3):
            for row in range(ROWS - 3):
                adjacent_pieces = [board[row+i][col+i] for i in range(4)]
                score += self.evaluate_adjacents(adjacent_pieces, player)

        # Check negatively sloped diagonals
        for col in range(COLUMNS - 3):
            for row in range(3, ROWS):
                adjacent_pieces = [board[row - i][col + i] for i in range(4)]
                score += self.evaluate_adjacents(adjacent_pieces, player)

        return score

    def evaluate_adjacents(self, adjacent_pieces, player):
        opponent = HUMAN if player == AI else AI
        player_pieces = adjacent_pieces.count(player)
        opponent_pieces = adjacent_pieces.count(opponent)
        empty_spaces = adjacent_pieces.count(EMPTY)

        if player_pieces == 4:
            return 1000
        elif player_pieces == 3 and empty_spaces == 1:
            return 50
        elif player_pieces == 2 and empty_spaces == 2:
            return 10

        # Penalize situations where the opponent is close to winning
        if opponent_pieces == 3 and empty_spaces == 1:
            return -80
        if opponent_pieces == 2 and empty_spaces == 2:
            return -20

        return 0