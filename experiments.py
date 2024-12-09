import numpy as np
import random
from utils import create_board, is_terminal_board, detect_win, valid_locations, place_piece, encode_state
from agents.qlearningAgent import ConnectFourQLearningAgent
from agents.minimaxAgent import MinimaxAgent
from agents.randomAgent import RandomAgent
from agents.learningAgents import AI, HUMAN

def play_game(agent1, agent2, verbose=False):
    """
    Simulates a single game between two agents.
    agent1: The first agent (plays as AI).
    agent2: The second agent (plays as HUMAN).
    verbose: If True, prints the game board after each move.
    
    Returns:
    - 1 if agent1 wins.
    - -1 if agent2 wins.
    - 0 for a draw.
    """
    board = create_board()
    current_player = AI  # Start with agent1
    agents = {AI: agent1, HUMAN: agent2}
    
    while not is_terminal_board(board):
        state = encode_state(board)
        action = agents[current_player].getAction(state)
        
        if action not in valid_locations(board):
            # If an invalid action is chosen, skip the turn (error handling)
            action = random.choice(valid_locations(board))
        
        place_piece(board, current_player, action)
        
        if verbose:
            print(f"Player {current_player} placed a piece in column {action}")
            print(board)
        
        if detect_win(board, current_player):
            return 1 if current_player == AI else -1  # Return winner
        
        current_player = HUMAN if current_player == AI else AI  # Switch player

    return 0  # Draw

def run_experiment(agent1, agent2, num_games=100, verbose=False):
    """
    Runs multiple games between two agents and calculates win rates.
    agent1: The first agent (plays as AI).
    agent2: The second agent (plays as HUMAN).
    num_games: Number of games to play.
    verbose: If True, prints game outcomes.
    
    Returns:
    - A dictionary with win rates for agent1, agent2, and draws.
    """
    results = {1: 0, -1: 0, 0: 0}  # Win (1), Loss (-1), Draw (0)

    for i in range(num_games):
        result = play_game(agent1, agent2, verbose)
        results[result] += 1
        if verbose:
            print(f"Game {i+1}/{num_games}: {'Agent1 Wins' if result == 1 else 'Agent2 Wins' if result == -1 else 'Draw'}")

    win_rate_agent1 = results[1] / num_games
    win_rate_agent2 = results[-1] / num_games
    draw_rate = results[0] / num_games

    return {
        "Agent1 Win Rate": win_rate_agent1,
        "Agent2 Win Rate": win_rate_agent2,
        "Draw Rate": draw_rate
    }

def main():
    # Define agents
    random_agent = RandomAgent()
    minimax_agent = MinimaxAgent(depth=10)
    qlearning_agent = ConnectFourQLearningAgent(numTraining=10000, epsilon=0.1, alpha=0.5, gamma=0.9)

    # Train the Q-learning agent
    #print("Training Q-Learning Agent...")
    #qlearning_agent.train()
    #print("Training complete!")

    # Experiment 1: Minimax vs Random
    print("Running Experiment 1: Minimax Agent vs Random Agent...")
    results1 = run_experiment(minimax_agent, random_agent, num_games=1000)
    print("Results:", results1)

    # Experiment 2: Q-Learning vs Random
    #print("Running Experiment 2: Q-Learning Agent vs Random Agent...")
    #results2 = run_experiment(qlearning_agent, random_agent, num_games=100)
    #print("Results:", results2)

    # Experiment 3: Q-Learning vs Minimax
    #print("Running Experiment 3: Q-Learning Agent vs Minimax Agent...")
    #results3 = run_experiment(qlearning_agent, minimax_agent, num_games=100)
    #print("Results:", results3)

if __name__ == "__main__":
    main()