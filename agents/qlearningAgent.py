from agents.minimaxAgent import MinimaxAgent
from agents.randomAgent import RandomAgent
import random
import numpy as np
from utils import valid_locations, encode_state, clone_and_place_piece, is_terminal_board, detect_win, place_piece, getStateRepresentation, create_empty_board, getLegalActions, HUMAN, AI, ROWS, COLUMNS, EMPTY

class ConnectFourQLearningAgent:
    def __init__(self, alpha=0.05, gamma=0.9, epsilon=0.1, numTraining=1000):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.numTraining = numTraining
        self.qvalues = {}
        self.trainingCompleted = False

    def getQValue(self, state, action):
        return self.qvalues.get((state, action), 0.0)

    def getValue(self, state):
        actions = getLegalActions(state)
        if len(actions) == 0:
            return 0.0
        values = [self.getQValue(state, a) for a in actions]
        return max(values)

    def getPolicy(self, state):
        actions = getLegalActions(state)
        if not actions:
            return None
        best_value = self.getValue(state)
        best_actions = [a for a in actions if self.getQValue(state, a) == best_value]
        return random.choice(best_actions)

    def getAction(self, board):
        state = getStateRepresentation(board)
        actions = getLegalActions(state)
        if len(actions) == 0:
            return None

        if self.trainingCompleted or random.random() > self.epsilon:
            return self.getPolicy(state)
        else:
            return random.choice(actions)

    def update(self, prevState, action, nextState, reward):
        old_value = self.getQValue(prevState, action)
        future_value = self.getValue(nextState)
        new_value = old_value + self.alpha * (reward + self.gamma * future_value - old_value)
        self.qvalues[(prevState, action)] = new_value

    def evaluate_intermediate_state(self, board):
        """
        Returns a small intermediate reward for a non-terminal board state.
        Positive values if AI is doing well (e.g. forming threats),
        Negative values if the opponent is forming threats.
        """

        score = 0.0
        AI = 2
        HUMAN = 1
        EMPTY = 0

        # Helper function to score a window of 4 cells
        def score_window(window):
            AI_count = np.count_nonzero(window == AI)
            HUMAN_count = np.count_nonzero(window == HUMAN)
            EMPTY_count = np.count_nonzero(window == EMPTY)

            # If both have pieces in this window, no direct threat
            if AI_count > 0 and HUMAN_count > 0:
                return 0

            # Potential threats for AI
            if AI_count == 3 and EMPTY_count == 1:
                return 0.01  # AI close to winning
            if AI_count == 2 and EMPTY_count == 2:
                return 0.005  # AI building towards a threat

            # Potential threats for HUMAN (opponent)
            if HUMAN_count == 3 and EMPTY_count == 1:
                return -10.0  # Opponent close to winning
            if HUMAN_count == 2 and EMPTY_count == 2:
                return -0.02

            return 0

        board_array = np.array(board)

        # Check all horizontal windows
        for r in range(ROWS):
            for c in range(COLUMNS - 3):
                window = board_array[r, c:c+4]
                score += score_window(window)

        # Check vertical windows
        for c in range(COLUMNS):
            for r in range(ROWS - 3):
                window = board_array[r:r+4, c]
                score += score_window(window)

        # Check positively sloped diagonals
        for r in range(ROWS - 3):
            for c in range(COLUMNS - 3):
                window = [board_array[r+i][c+i] for i in range(4)]
                score += score_window(window)

        # Check negatively sloped diagonals
        for r in range(ROWS - 3):
            for c in range(3, COLUMNS):
                window = [board_array[r+i][c-i] for i in range(4)]
                score += score_window(window)

        return score

    def train(self, numEpisodes=10000, switch_episode=-1):
        random_opponent = RandomAgent()
        minimax_opponent = MinimaxAgent(depth=2)

        print("EMPTY:", EMPTY, "HUMAN:", HUMAN, "AI:", AI)
        print(f"Starting Q-Learning training for {numEpisodes} episodes.")

        initial_epsilon = 1.0
        decay_rate = 0.9995
        epsilon_min = 0.05

        wins = 0
        losses = 0
        draws = 0

        block_wins = 0
        block_losses = 0
        block_draws = 0

        for episode in range(numEpisodes):
            if episode < switch_episode:
                opponent = random_opponent
            else:
                opponent = minimax_opponent

            board = create_empty_board()
            current_player = HUMAN
            game_over = False
            state = getStateRepresentation(board)
            prev_state = None
            prev_action = None

            # Epsilon decay
            self.epsilon = max(initial_epsilon * (decay_rate ** episode), epsilon_min)

            episode_result = None

            while not game_over:
                if current_player == AI:
                    actions = getLegalActions(state)
                    if not actions:
                        game_over = True
                        episode_result = 'draw'
                        continue

                    action = self.getAction(board)
                    if action not in actions:
                        action = random.choice(actions)

                    prev_state, prev_action = state, action
                    board = clone_and_place_piece(board, AI, action)
                else:
                    actions = valid_locations(board)
                    if not actions:
                        game_over = True
                        episode_result = 'draw'
                        continue

                    action = opponent.getAction(board)
                    if action not in actions:
                        action = random.choice(actions)

                    board = clone_and_place_piece(board, HUMAN, action)

                state = getStateRepresentation(board)

                if is_terminal_board(board):
                    game_over = True
                    if detect_win(board, AI):
                        episode_result = 'win'
                    elif detect_win(board, HUMAN):
                        episode_result = 'loss'
                    else:
                        episode_result = 'draw'

                # If the game ended, update with terminal reward
                if game_over and prev_state is not None and prev_action is not None:
                    if episode_result == 'win':
                        reward = 10.0
                    elif episode_result == 'loss':
                        reward = -10.0
                    else:
                        reward = 0.0
                    self.update(prev_state, prev_action, state, reward)
                else:
                    # Non-terminal: provide intermediate reward
                    # Only for the AI's move (when prev_state and prev_action are set)
                    if prev_state is not None and prev_action is not None and current_player == AI:
                        intermediate_reward = self.evaluate_intermediate_state(board)
                        self.update(prev_state, prev_action, state, intermediate_reward)

                current_player = HUMAN if current_player == AI else AI

            if episode_result == 'win':
                wins += 1
                block_wins += 1
            elif episode_result == 'loss':
                losses += 1
                block_losses += 1
            elif episode_result == 'draw':
                draws += 1
                block_draws += 1

            if (episode + 1) % (numEpisodes / 10) == 0:
                if hasattr(self, 'prev_qvalues'):
                    diff = 0.0
                    all_keys = set(self.qvalues.keys()).union(set(self.prev_qvalues.keys()))
                    for k in all_keys:
                        old_val = self.prev_qvalues.get(k, 0.0)
                        new_val = self.qvalues.get(k, 0.0)
                        diff += abs(new_val - old_val)
                    print(f"Total Q-value difference since last checkpoint: {diff:.4f}")
                self.prev_qvalues = self.qvalues.copy()

                block_total = block_wins + block_losses + block_draws
                if block_total > 0:
                    block_win_rate = block_wins / block_total
                    print(f"Win Rate for episodes {episode-(block_total-1)}-{episode+1}: {block_win_rate:.2f} "
                        f"(W:{block_wins}, L:{block_losses}, D:{block_draws})")
                else:
                    print("No completed episodes in the last block, cannot compute block win rate.")

                block_wins = 0
                block_losses = 0
                block_draws = 0

        self.epsilon = 0.0
        self.trainingCompleted = True
        print("Training completed. Epsilon set to 0. The agent will now act greedily.")

