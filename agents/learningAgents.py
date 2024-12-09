import time
import numpy as np
from utils import raiseNotDefined, encode_state, ROWS, COLUMNS, AI, HUMAN, EMPTY

class Agent:
    def __init__(self, index=0):
        self.index = index

    def getAction(self, state):
        raiseNotDefined()

class ValueEstimationAgent(Agent):
    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, numTraining=10):
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(numTraining)

    def getQValue(self, state, action):
        raise NotImplementedError

    def getValue(self, state):
        raise NotImplementedError

    def getPolicy(self, state):
        raise NotImplementedError

    def getAction(self, state):
        raise NotImplementedError

class ReinforcementAgent(ValueEstimationAgent):
    def __init__(self, actionFn=None, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1):
        if actionFn is None:
            actionFn = lambda state: state.getLegalActions()  # placeholder, replaced by environment-specific logic
        self.actionFn = actionFn
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        ValueEstimationAgent.__init__(self, alpha=alpha, epsilon=epsilon, gamma=gamma, numTraining=numTraining)

    def update(self, state, action, nextState, reward):
        raise NotImplementedError

    def getLegalActions(self, state):
        return self.actionFn(state)

    def observeTransition(self, state, action, nextState, deltaReward):
        self.episodeRewards += deltaReward
        self.update(state, action, nextState, deltaReward)

    def startEpisode(self):
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            self.epsilon = 0.0
            self.alpha = 0.0

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setLearningRate(self, alpha):
        self.alpha = alpha

    def setDiscount(self, discount):
        self.discount = discount

    def doAction(self, state, action):
        self.lastState = state
        self.lastAction = action

    '''
    def observationFunction(self, state):
        # This would be called by the environment after each step,
        # in Connect Four, you must adapt this to your environment states.
        if self.lastState is not None:
            reward = state.getScore() - self.lastState.getScore()
            self.observeTransition(self.lastState, self.lastAction, state, reward)
        return state

    def registerInitialState(self, state):
        self.startEpisode()
        if self.episodesSoFar == 0:
            print('Beginning %d episodes of Training' % (self.numTraining))

    def final(self, state):
        deltaReward = state.getScore() - self.lastState.getScore()
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()
        # You can add the printing of statistics here if needed.
    '''
    def registerInitialState(self, board):
        # Called at the start of an episode
        # board is a np array representing the initial board
        state = encode_state(board)
        self.startEpisode()
        if self.episodesSoFar == 0:
            print('Beginning %d episodes of Training' % (self.numTraining))
        return state

    def observationFunction(self, board):
        # Called after each action/transition by the environment
        # Convert board to state
        state = encode_state(board)
        if self.lastState is not None:
            # We don't have a traditional state.getScore(), but we can define a heuristic score function
            reward = self.stateScore(board) - self.stateScore(self.decode_state(self.lastState))
            self.observeTransition(self.lastState, self.lastAction, state, reward)
        return state

    def final(self, board):
        # Called at the end of an episode
        state = encode_state(board)
        deltaReward = self.stateScore(board) - self.stateScore(self.decode_state(self.lastState))
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

    def decode_state(self, state):
        # Convert tuple back to board
        return np.array(state).reshape((ROWS, COLUMNS))

    def stateScore(self, board):
        # Define a heuristic scoring for the board to determine intermediate rewards
        ai_score = self.score(board, AI)
        human_score = self.score(board, HUMAN)
        return ai_score - human_score
    
    def stateScore(self, board):
        """
        Computes a heuristic score for the board.
        """
        def evaluate_window(window, player):
            """
            Evaluates a 4-cell window for scoring.
            """
            score = 0
            opponent = HUMAN if player == AI else AI

            player_count = np.count_nonzero(window == player)
            opponent_count = np.count_nonzero(window == opponent)
            empty_count = np.count_nonzero(window == EMPTY)  # Assuming 0 is EMPTY

            if player_count == 4:
                score += 100  # Win
            elif player_count == 3 and empty_count == 1:
                score += 5  # Potential win
            elif player_count == 2 and empty_count == 2:
                score += 2  # Potential setup

            if opponent_count == 3 and empty_count == 1:
                score -= 4  # Block opponent

            return score

        score = 0
        # Center column preference
        center_col = COLUMNS // 2
        center_array = [board[row][center_col] for row in range(ROWS)]
        score += center_array.count(AI) * 3

        # Horizontal, vertical, and diagonal windows
        for row in range(ROWS):
            for col in range(COLUMNS - 3):
                score += evaluate_window(board[row, col:col + 4], AI)

        for col in range(COLUMNS):
            for row in range(ROWS - 3):
                window = [board[row + i][col] for i in range(4)]
                score += evaluate_window(window, AI)

        for row in range(ROWS - 3):
            for col in range(COLUMNS - 3):
                window = [board[row + i][col + i] for i in range(4)]
                score += evaluate_window(window, AI)

            for col in range(3, COLUMNS):
                window = [board[row + i][col - i] for i in range(4)]
                score += evaluate_window(window, AI)

        return score