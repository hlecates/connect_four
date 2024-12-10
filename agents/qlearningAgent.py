from agents.minimaxAgent import MinimaxAgent
from agents.randomAgent import RandomAgent
import random
import numpy as np
import statistics  # for variance calculation, or you can use numpy if preferred
from utils import valid_locations, encode_state, clone_and_place_piece, is_terminal_board, detect_win, place_piece, HUMAN, AI, ROWS, COLUMNS, EMPTY

class ConnectFourQLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, numTraining=1000):
        """
        Initialize the Q-Learning agent.

        Args:
            alpha (float): learning rate
            gamma (float): discount factor
            epsilon (float): epsilon for epsilon-greedy policy
            numTraining (int): number of training episodes
            opponent: an opponent agent to train against
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.numTraining = numTraining

        # Q-values: keys are (state, action), values are floats
        self.qvalues = {}
        self.trainingCompleted = False

    def getQValue(self, state, action):
        """
        Returns the Q-value for (state, action).
        If no value is stored, defaults to 0.0.
        """
        return self.qvalues.get((state, action), 0.0)

    def getValue(self, state):
        """
        Returns the maximum Q-value for any action in the given state.
        If there are no valid moves, returns 0.0.
        """
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return 0.0

        values = [self.getQValue(state, a) for a in actions]
        return max(values)

    def getPolicy(self, state):
        """
        Returns the best action (policy) for a given state according to the current Q-values.
        If multiple actions have the same value, picks one randomly.
        """
        actions = self.getLegalActions(state)
        if not actions:
            return None

        best_value = self.getValue(state)
        # actions with best Q-value
        best_actions = [a for a in actions if self.getQValue(state, a) == best_value]
        return random.choice(best_actions)

    def getAction(self, board):
        """
        Chooses an action for the current board state.
        If training is completed, always choose the best action.
        Otherwise, use epsilon-greedy strategy.
        """
        state = self.getStateRepresentation(board)
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return None

        # If training completed, behave greedily
        if self.trainingCompleted or random.random() > self.epsilon:
            return self.getPolicy(state)
        else:
            return random.choice(actions)

    def update(self, prevState, action, nextState, reward):
        """
        Update Q-values based on the transition:
        Q(s,a) <- Q(s,a) + alpha [r + gamma*max_a' Q(s', a') - Q(s,a)]
        """
        old_value = self.getQValue(prevState, action)
        future_value = self.getValue(nextState)
        new_value = old_value + self.alpha * (reward + self.gamma * future_value - old_value)
        self.qvalues[(prevState, action)] = new_value

    def getLegalActions(self, state):
        """
        Given a state representation, return all valid actions (columns) where a piece can be placed.
        """
        # State is a representation of the board, decode if needed
        # We'll rely on the global board shape and knowledge that
        # the board is a ROWS x COLUMNS matrix.
        board = self.decode_state(state)
        return valid_locations(board)

    def getStateRepresentation(self, board):
        """
        Encode the board as a tuple/string for dictionary keys in Q-table.
        """
        return encode_state(board)

    def decode_state(self, state):
        board_array = np.array(state).reshape((ROWS, COLUMNS))
        return board_array

    def evaluate_intermediate_state(self, board):
        """
        Evaluate the board to provide an intermediate reward.
        The goal is to return a small but informative value:
        - Positive if the AI is in a good position.
        - Negative if the opponent is in a good position.
        """

        board_array = np.array(board)
        score = 0
        
        # Define constants for scoring patterns
        THREE_IN_A_ROW_AI = 5
        TWO_IN_A_ROW_AI = 2
        THREE_IN_A_ROW_OPP = -1000000
        TWO_IN_A_ROW_OPP = -5
        
        # Helper to evaluate a 4-cell window
        def evaluate_window(window):
            AI_count = np.count_nonzero(window == AI)
            HUMAN_count = np.count_nonzero(window == HUMAN)
            EMPTY_count = np.count_nonzero(window == EMPTY)

            # If both players have pieces in this window, it's not useful for either
            if AI_count > 0 and HUMAN_count > 0:
                return 0

            # Scoring logic:
            # Prioritize AI making "threats"
            if AI_count == 3 and EMPTY_count == 1:
                return THREE_IN_A_ROW_AI
            elif AI_count == 2 and EMPTY_count == 2:
                return TWO_IN_A_ROW_AI
            
            # Penalize opponent threats
            if HUMAN_count == 3 and EMPTY_count == 1:
                return THREE_IN_A_ROW_OPP
            elif HUMAN_count == 2 and EMPTY_count == 2:
                return TWO_IN_A_ROW_OPP

            return 0
        
        # Center column preference (gives more flexibility)
        center_col = COLUMNS // 2
        center_array = board_array[:, center_col]
        center_count = np.count_nonzero(center_array == AI)
        score += center_count * 3  # Small bonus for playing in center

        # Evaluate horizontal windows
        for r in range(ROWS):
            for c in range(COLUMNS - 3):
                window = board_array[r, c:c+4]
                score += evaluate_window(window)

        # Evaluate vertical windows
        for c in range(COLUMNS):
            for r in range(ROWS - 3):
                window = board_array[r:r+4, c]
                score += evaluate_window(window)

        # Evaluate positively sloped diagonals
        for r in range(ROWS - 3):
            for c in range(COLUMNS - 3):
                window = [board_array[r+i][c+i] for i in range(4)]
                score += evaluate_window(window)

        # Evaluate negatively sloped diagonals
        for r in range(ROWS - 3):
            for c in range(3, COLUMNS):
                window = [board_array[r+i][c-i] for i in range(4)]
                score += evaluate_window(window)

        baseline = 0.01
        scaling_factor = 0.1 
        return baseline + scaling_factor * score

    

    def train(self, numEpisodes=10000, switch_episode=-1):
        # Create opponents
        random_opponent = RandomAgent()
        minimax_opponent = MinimaxAgent(depth=3)  # Adjust depth as needed

        print("EMPTY:", EMPTY, "HUMAN:", HUMAN, "AI:", AI)
        print(f"Starting Q-Learning training for {numEpisodes} episodes.")

        initial_epsilon = 1.0
        decay_rate = 0.9995

        # Tracking cumulative outcomes
        wins = 0
        losses = 0
        draws = 0

        # Tracking outcomes for the last 1000 episodes
        block_wins = 0
        block_losses = 0
        block_draws = 0

        for episode in range(numEpisodes):
            # Decide which opponent to face this episode
            if episode < switch_episode:
                opponent = random_opponent
            else:
                opponent = minimax_opponent

            board = self.create_empty_board()
            current_player = HUMAN
            game_over = False
            state = self.getStateRepresentation(board)
            prev_state = None
            prev_action = None

            # Apply epsilon decay
            self.epsilon = initial_epsilon * (decay_rate ** episode)

            episode_result = None

            while not game_over:
                if current_player == AI:
                    actions = self.getLegalActions(state)
                    if not actions:
                        game_over = True
                        episode_result = 'draw'
                        continue

                    action = self.getAction(board)
                    if action not in actions:
                        # Debug: Log invalid action
                        print(f"Invalid AI action selected: {action}")
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
                        print(f"Invalid opponent action selected: {action}")
                        action = random.choice(actions)

                    board = clone_and_place_piece(board, HUMAN, action)

                state = self.getStateRepresentation(board)

                if is_terminal_board(board):
                    game_over = True
                    if detect_win(board, AI):
                        episode_result = 'win'
                    elif detect_win(board, HUMAN):
                        episode_result = 'loss'
                    else:
                        episode_result = 'draw'

                # Update Q-values
                if game_over and prev_state is not None and prev_action is not None:
                    if episode_result == 'win':
                        reward = 10.0
                    elif episode_result == 'loss':
                        reward = -10.0
                    else:
                        reward = 0.5
                    self.update(prev_state, prev_action, state, reward)
                elif prev_state is not None and prev_action is not None:
                    intermediate_reward = self.evaluate_intermediate_state(board)
                    self.update(prev_state, prev_action, state, intermediate_reward)

                current_player = HUMAN if current_player == AI else AI

            # Update cumulative and block-level counters
            if episode_result == 'win':
                wins += 1
                block_wins += 1
            elif episode_result == 'loss':
                losses += 1
                block_losses += 1
            elif episode_result == 'draw':
                draws += 1
                block_draws += 1

            # Every 1000 episodes, track Q-value differences, print segment win rate and reset block counters
            if (episode + 1) % 1000 == 0:
                if hasattr(self, 'prev_qvalues'):
                    diff = 0.0
                    all_keys = set(self.qvalues.keys()).union(set(self.prev_qvalues.keys()))
                    for k in all_keys:
                        old_val = self.prev_qvalues.get(k, 0.0)
                        new_val = self.qvalues.get(k, 0.0)
                        diff += abs(new_val - old_val)
                    print(f"Total Q-value difference since last checkpoint: {diff:.4f}")
                self.prev_qvalues = self.qvalues.copy()

                # Compute win rate for the last 1000 episodes
                block_total = block_wins + block_losses + block_draws
                if block_total > 0:
                    block_win_rate = block_wins / block_total
                    print(f"Win Rate for episodes {episode-999}-{episode+1}: {block_win_rate:.2f} "
                        f"(W:{block_wins}, L:{block_losses}, D:{block_draws})")
                else:
                    print(f"No completed episodes in the last 1000 games, cannot compute block win rate.")

                # Reset block counters
                block_wins = 0
                block_losses = 0
                block_draws = 0

        self.epsilon = 0.0
        self.trainingCompleted = True
        print("Training completed. Epsilon set to 0. The agent will now act greedily.")


    def create_empty_board(self):
        return np.zeros((ROWS, COLUMNS), dtype = np.int8)