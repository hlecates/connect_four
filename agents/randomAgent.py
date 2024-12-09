import random
from utils import valid_locations, EMPTY

class RandomAgent:
    """
    An agent that selects a random valid move.
    """
    def getAction(self, board):
        """
        Choose a random valid action (column) from the current board state.

        Args:
            board (np.ndarray): The current board state.

        Returns:
            int: The chosen column to place the piece.
        """
        valid_moves = valid_locations(board)
        if not valid_moves:
            return None
        return random.choice(valid_moves)