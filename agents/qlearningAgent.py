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