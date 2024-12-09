'''
import random
from agents.minimaxAgent import MinimaxAgent
from agents.learningAgents import ReinforcementAgent
from agents.randomAgent import RandomAgent
from utils import Counter, flipCoin, np, ROWS, COLUMNS, valid_locations, create_board, is_terminal_board, detect_win, place_piece, encode_state, AI, HUMAN, EMPTY

class QLearningAgent(ReinforcementAgent):
    """
    A general Q-Learning Agent, environment-agnostic.
    """
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.Q_vals = Counter()

    def getQValue(self, state, action):
        return self.Q_vals[(state, action)]

    def computeValueFromQValues(self, state):
        actions = self.getLegalActions(state)
        if not actions:
            return 0.0
        return max(self.getQValue(state, a) for a in actions)

    def computeActionFromQValues(self, state):
        actions = self.getLegalActions(state)
        if not actions:
            return None
        maxQ = self.computeValueFromQValues(state)
        bestActions = [a for a in actions if self.getQValue(state, a) == maxQ]
        if not bestActions:
            return None
        return random.choice(bestActions)

    def getAction(self, state):
        actions = self.getLegalActions(state)
        if not actions:
            return None

        if flipCoin(self.epsilon):
            # Exploration
            action = random.choice(actions)
        else:
            # Exploitation
            action = self.computeActionFromQValues(state)

        self.doAction(state, action)
        return action

    def update(self, state, action, nextState, reward):
        currentQ = self.getQValue(state, action)
        maxNextQ = self.computeValueFromQValues(nextState)
        updatedQ = currentQ + self.alpha * (reward + self.discount * maxNextQ - currentQ)
        self.Q_vals[(state, action)] = updatedQ

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class ConnectFourQLearningAgent(QLearningAgent):
    """
    A Q-Learning Agent adapted for Connect Four with improved reward structure
    and epsilon decay schedule.
    """

    def getLegalActions(self, state):
        # State is a tuple representing the board
        board = np.array(state).reshape((ROWS, COLUMNS))
        return valid_locations(board)

    def getAction(self, state):
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action

    def train(self, numRandomEpisodes=5000, switch_episode=80000):
        """
        Train the Q-learning agent by playing against RandomAgent first,
        then MinimaxAgent after 'switch_episode' episodes.

        Adjusted reward structure and epsilon decay.
        """
        print("Training is beginning...")
        total_episodes = self.numTraining
        print_interval = max(1, total_episodes // 10)  # Print progress every 10% of training

        # Opponents
        random_opponent = RandomAgent()
        minimax_opponent = MinimaxAgent(depth=1)

        for episode in range(1, total_episodes + 1):
            board = create_board()
            self.startEpisode()
            current_player = AI
            state = encode_state(board)

            # More gradual epsilon decay:
            # start with epsilon ~0.4, decay towards ~0.01 by end of training
            # self.epsilon = max(0.01, 0.4 * (1 - (episode / float(total_episodes))))
            self.epsilon = 1 / episode

            lastState = None
            lastAction = None

            # Choose opponent
            if episode <= switch_episode:
                opponent = random_opponent
            else:
                opponent = minimax_opponent

            while not is_terminal_board(board):
                if current_player == AI:
                    # Q-agent's turn
                    action = self.getAction(state)
                    if action is None:
                        # No moves (terminal)
                        break
                    place_piece(board, AI, action)
                    nextState = encode_state(board)

                    # Compute reward
                    if detect_win(board, AI):
                        reward = 100.0  # Large positive reward for winning
                    elif is_terminal_board(board):
                        # Draw is neutral or slightly negative to encourage winning
                        reward = -10.0
                    else:
                        # Intermediate reward from a heuristic evaluation
                        reward = self.stateScore(board) * 0.5

                    self.update(state, action, nextState, reward)
                    state = nextState
                    lastState, lastAction = state, action
                    current_player = HUMAN

                else:
                    # Opponent's turn
                    valid_moves = valid_locations(board)
                    if not valid_moves:
                        break
                    opp_col = opponent.getAction(board)
                    if opp_col not in valid_moves:
                        opp_col = random.choice(valid_moves)
                    place_piece(board, HUMAN, opp_col)
                    state = encode_state(board)

                    # If opponent wins, strong negative reward for AI's last move
                    if detect_win(board, HUMAN) and lastState is not None and lastAction is not None:
                        self.update(lastState, lastAction, state, -100.0)

                    current_player = AI

            self.stopEpisode()
            if episode % print_interval == 0 or episode == total_episodes:
                print(f"Training progress: {episode}/{total_episodes} episodes. Epsilon: {self.epsilon:.4f}")

        print("Training has finished!")

    def stateScore(self, board):
        """
        Compute a heuristic score for the current board from the AI's perspective.
        We look at all possible 4-cell windows and score them.

        Scoring logic:
          - Four in a row (AI): large reward (captured by terminal reward above)
          - Three in a row with an open space: moderate positive
          - Two in a row with open spaces: small positive
          - Opponent threats: negative

        This scoring is scaled down since terminal conditions give the big rewards.
        """
        def evaluate_window(window, player):
            opponent = HUMAN if player == AI else AI
            player_count = np.count_nonzero(window == player)
            opp_count = np.count_nonzero(window == opponent)
            empty_count = np.count_nonzero(window == EMPTY)

            score = 0
            # Prioritize potential wins
            if player_count == 3 and empty_count == 1:
                score += 10
            elif player_count == 2 and empty_count == 2:
                score += 5

            # Penalize opponent threats
            if opp_count == 3 and empty_count == 1:
                score -= 20
            return score

        score = 0
        # Center column preference
        center_col = COLUMNS // 2
        center_array = [board[r][center_col] for r in range(ROWS)]
        center_count = center_array.count(AI)
        score += center_count * 3

        # Horizontal windows
        for r in range(ROWS):
            for c in range(COLUMNS - 3):
                window = board[r, c:c+4]
                score += evaluate_window(window, AI)

        # Vertical windows
        for c in range(COLUMNS):
            for r in range(ROWS - 3):
                window = board[r:r+4, c]
                score += evaluate_window(window, AI)

        # Positive diagonal windows
        for r in range(ROWS - 3):
            for c in range(COLUMNS - 3):
                window = [board[r+i][c+i] for i in range(4)]
                score += evaluate_window(window, AI)

        # Negative diagonal windows
        for r in range(ROWS - 3):
            for c in range(3, COLUMNS):
                window = [board[r+i][c-i] for i in range(4)]
                score += evaluate_window(window, AI)

        return score
'''

from agents.minimaxAgent import MinimaxAgent
from agents.randomAgent import RandomAgent
import random
import math
from utils import valid_locations, encode_state, clone_and_place_piece, is_terminal_board, detect_win, HUMAN, AI, ROWS, COLUMNS, EMPTY

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

    def train(self, opponent):
        if opponent is None:
            raise ValueError("No opponent provided for training!")

        print(f"Starting Q-Learning training for {self.numTraining} episodes against {opponent.__class__.__name__}.")

        for episode in range(self.numTraining):
            board = self.create_empty_board()
            current_player = HUMAN  # Human = first player, AI = second player in this code.
            game_over = False
            state = self.getStateRepresentation(board)
            prev_state = None
            prev_action = None

            self.epsilon = 1 / (episode + 1)

            # Debug: Print progress
            if (episode + 1) % 1000 == 0 or (episode + 1) == self.numTraining:
                print(f"Episode: {episode+1}/{self.numTraining}, Epsilon: {self.epsilon}")

            move_count = 0  # Count how many moves are made in this episode for debugging

            while not game_over:
                # Debug: Print current board state every 10000 episodes to track progress
                if (episode + 1) % 10000 == 0 and move_count < 5:  
                    # Only print a few moves per episode to avoid huge logs
                    print(f"Episode {episode+1}, Move {move_count}:")
                    for r in board:
                        print(r)
                    print("-----------------")

                if current_player == AI:
                    actions = self.getLegalActions(state)
                    # Debug: Check what actions are available
                    if (episode + 1) % 10000 == 0 and move_count < 5:
                        print("AI Turn. Available actions:", actions)

                    action = self.getAction(board)
                    if action is None:
                        # No moves available
                        game_over = True
                        # Debug:
                        if (episode + 1) % 10000 == 0:
                            print("No moves available for AI, game over.")
                    else:
                        # Agent takes action
                        new_board = clone_and_place_piece(board, AI, action)
                        prev_state, prev_action = state, action
                        board = new_board
                else:
                    # Opponent's turn
                    actions = valid_locations(board)
                    # Debug: Check opponent moves
                    if (episode + 1) % 10000 == 0 and move_count < 5:
                        print(f"Opponent Turn. Available actions for {opponent.__class__.__name__}: {actions}")

                    action = opponent.getAction(board)
                    if action is None:
                        # Opponent has no moves
                        game_over = True
                        # Debug:
                        if (episode + 1) % 10000 == 0:
                            print("Opponent has no moves, game over.")
                    else:
                        board = clone_and_place_piece(board, HUMAN, action)

                state = self.getStateRepresentation(board)
                move_count += 1

                # Debug: Check if the board is terminal
                if is_terminal_board(board):
                    # Debug:
                    if (episode + 1) % 10000 == 0:
                        print("Board is terminal")
                    game_over = True

                # If game ended, assign reward
                if game_over and prev_state is not None and prev_action is not None:
                    if detect_win(board, AI):
                        reward = 10.0
                        print("AI wins this episode!")
                    elif detect_win(board, HUMAN):
                        reward = -10.0
                        print(f"{opponent.__class__.__name__} wins this episode!")
                    else:
                        reward = 0.5
                        print("This episode ended in a draw.")
                    self.update(prev_state, prev_action, state, reward)
                
                elif prev_state is not None and prev_action is not None:
                    # Non-terminal move: update with 0 reward
                    self.update(prev_state, prev_action, state, 0)

                # Switch player
                current_player = HUMAN if current_player == AI else AI

        # After training
        self.epsilon = 0.0
        self.trainingCompleted = True
        print("Training completed. Epsilon set to 0. The agent will now act greedily.")



    def create_empty_board(self):
        """
        Create an empty board represented as a 2D list.
        """
        board = [[EMPTY for _ in range(COLUMNS)] for _ in range(ROWS)]
        return board
