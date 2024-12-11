import math
import random
from utils import valid_locations, clone_and_place_piece, detect_win, is_terminal_board, AI, HUMAN, COLUMNS, ROWS, EMPTY

class MinimaxAgent:
    def __init__(self, depth=5, player=AI):
        """
        player: which player this agent is representing (AI=2 or HUMAN=1).
                The agent will try to maximize the outcome for 'player'.
        depth: search depth for minimax
        """
        self.depth = depth
        self.player = player
        self.opponent = HUMAN if player == AI else AI

    def getAction(self, board):
        col, _ = self.maxValue(board, self.depth, -math.inf, math.inf)
        return col

    def maxValue(self, board, ply, alpha, beta):
        # maxValue: from self.player's perspective, we are trying to maximize self.player's value
        terminal = is_terminal_board(board)
        if ply == 0 or terminal:
            return self.terminal_evaluation(board)

        valid_cols = valid_locations(board)
        if not valid_cols:
            # No moves left, terminal draw
            return None, 0

        value = -math.inf
        col_choice = random.choice(valid_cols)

        # When maximizing, we place the current agent's pieces
        for c in valid_cols:
            next_board = clone_and_place_piece(board, self.player, c)
            _, new_score = self.minValue(next_board, ply-1, alpha, beta)
            if new_score > value:
                value = new_score
                col_choice = c
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return col_choice, value

    def minValue(self, board, ply, alpha, beta):
        # minValue: from self.player's perspective, this simulates the opponent's turn.
        # The opponent will try to minimize self.player's value.
        terminal = is_terminal_board(board)
        if ply == 0 or terminal:
            return self.terminal_evaluation(board)

        valid_cols = valid_locations(board)
        if not valid_cols:
            # No moves left, terminal draw
            return None, 0

        value = math.inf
        col_choice = random.choice(valid_cols)

        # When minimizing, we place the opponent's pieces
        for c in valid_cols:
            next_board = clone_and_place_piece(board, self.opponent, c)
            _, new_score = self.maxValue(next_board, ply-1, alpha, beta)
            if new_score < value:
                value = new_score
                col_choice = c
            beta = min(beta, value)
            if beta <= alpha:
                break
        return col_choice, value

    def terminal_evaluation(self, board):
        terminal = is_terminal_board(board)
        if terminal:
            if detect_win(board, self.opponent):
                # Opponent win is bad for self.player
                return (None, -1000000000)
            elif detect_win(board, self.player):
                # Self win is great
                return (None, 1000000000)
            else:
                # Draw
                return (None, 0)
        else:
            # Depth reached, evaluate board
            return (None, self.evaluate_board(board))

    def evaluate_board(self, board):
        # Score the board from the perspective of self.player
        return self.score(board, self.player)

    def score(self, board, player):
        score = 0
        center_col = COLUMNS // 2

        # Center column preference
        center_array = [board[r][center_col] for r in range(ROWS)]
        center_count = center_array.count(player)
        score += center_count * 4

        # Check all windows of length 4
        for row in range(ROWS):
            for col in range(COLUMNS - 3):
                window = [board[row][col+i] for i in range(4)]
                score += self.evaluate_adjacents(window, player)

        for col in range(COLUMNS):
            for row in range(ROWS - 3):
                window = [board[row+i][col] for i in range(4)]
                score += self.evaluate_adjacents(window, player)

        for row in range(ROWS - 3):
            for col in range(COLUMNS - 3):
                window = [board[row+i][col+i] for i in range(4)]
                score += self.evaluate_adjacents(window, player)

        for row in range(3, ROWS):
            for col in range(COLUMNS - 3):
                window = [board[row - i][col + i] for i in range(4)]
                score += self.evaluate_adjacents(window, player)

        return score

    def evaluate_adjacents(self, adjacent_pieces, player):
        opponent = HUMAN if player == AI else AI
        player_pieces = adjacent_pieces.count(player)
        opponent_pieces = adjacent_pieces.count(opponent)
        empty_spaces = adjacent_pieces.count(EMPTY)

        if player_pieces > 0 and opponent_pieces > 0:
            return 0

        if opponent_pieces == 0:
            if player_pieces == 4:
                return 100000
            if player_pieces == 3 and empty_spaces == 1:
                return 100
            if player_pieces == 2 and empty_spaces == 2:
                return 10
            if player_pieces == 1 and empty_spaces == 3:
                return 1

        if player_pieces == 0:
            if opponent_pieces == 3 and empty_spaces == 1:
                return -1000
            if opponent_pieces == 2 and empty_spaces == 2:
                return -50
            if opponent_pieces == 1 and empty_spaces == 3:
                return -2

        return 0
