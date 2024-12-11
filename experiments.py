import os
import numpy as np
from agents.minimaxAgent import MinimaxAgent
from agents.randomAgent import RandomAgent
from agents.qlearningAgent import ConnectFourQLearningAgent
from utils import create_empty_board, is_terminal_board, detect_win, valid_locations, place_piece, AI, HUMAN

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

def run_games(agent1, agent2, num_games=100):
    """
    Runs a specified number of games between agent1 (AI) and agent2 (HUMAN or another AI).
    Returns:
        wins_agent1, wins_agent2, draws, game_results_list
    where game_results_list is a list of results for each game ("win_agent1", "win_agent2", or "draw").
    """
    wins_agent1 = 0
    wins_agent2 = 0
    draws = 0
    game_results = []

    for _ in range(num_games):
        board = create_empty_board()
        #current_player = HUMAN  # Let's alternate who goes first each game if desired
        current_player = HUMAN if _ % 2 == 0 else AI
        # For simplicity, always start with HUMAN = 1, followed by AI = 2

        game_over = False

        while not game_over:
            if current_player == HUMAN:
                col = agent2.getAction(board)
                if col is None or col not in valid_locations(board):
                    # If no action, draw
                    game_over = True
                    result = 'draw'
                else:
                    place_piece(board, HUMAN, col)

            else:  # AI turn
                col = agent1.getAction(board)
                if col is None or col not in valid_locations(board):
                    game_over = True
                    result = 'draw'
                else:
                    place_piece(board, AI, col)

            if is_terminal_board(board):
                game_over = True
                if detect_win(board, AI):
                    result = 'win_agent1'
                elif detect_win(board, HUMAN):
                    result = 'win_agent2'
                else:
                    result = 'draw'

            current_player = HUMAN if current_player == AI else AI

        game_results.append(result)

        if result == 'win_agent1':
            wins_agent1 += 1
        elif result == 'win_agent2':
            wins_agent2 += 1
        else:
            draws += 1

    return wins_agent1, wins_agent2, draws, game_results

def record_results(filename, description, results, game_results=None):
    """
    Write the results of an experiment to a data file.
    `results` is a dict with keys/values for wins, losses, and draws.
    If game_results (list) is provided, also record each game's outcome.
    """
    with open(os.path.join('data', filename), 'w') as f:
        f.write(description + "\n")
        f.write("Results:\n")
        for k,v in results.items():
            f.write(f"{k}: {v}\n")

        if game_results is not None:
            f.write("\nPer-game outcomes:\n")
            for i, r in enumerate(game_results):
                f.write(f"Game {i+1}: {r}\n")

# ---------------------------
# Experiments
# ---------------------------

# 1. Minimax vs Minimax with various depths
minimax_depths = [1, 2, 3, 4, 5]
for d1 in minimax_depths:
    for d2 in minimax_depths:
        if d1 == d2:
            continue
        # Agent1 acts as AI (player=2)
        agent1 = MinimaxAgent(depth=d1, player=AI)
        # Agent2 acts as HUMAN (player=1)
        agent2 = MinimaxAgent(depth=d2, player=HUMAN)

        w1, w2, dr, game_res = run_games(agent1, agent2, num_games=100)
        total = w1 + w2 + dr
        results = {
            "Minimax_Depth_Agent1": d1,
            "Minimax_Depth_Agent2": d2,
            "Wins_Agent1": w1,
            "Wins_Agent2": w2,
            "Draws": dr,
            "Agent1_Win_Rate": w1/total if total > 0 else 0,
            "Agent2_Win_Rate": w2/total if total > 0 else 0,
            "Draw_Rate": dr/total if total > 0 else 0
        }
        filename = f"minimax_vs_minimax_d{d1}_d{d2}.txt"
        desc = f"Minimax vs Minimax: Agent1 depth={d1}, Agent2 depth={d2}"
        record_results(filename, desc, results, game_res)

# 2. Minimax vs Random
agent_minimax = MinimaxAgent(depth=3)
agent_random = RandomAgent()
w1, w2, dr, game_res = run_games(agent_minimax, agent_random, num_games=100)
total = w1 + w2 + dr
results = {
    "Minimax_Depth": 3,
    "Opponent": "Random",
    "Wins_MiniMax": w1,
    "Wins_Random": w2,
    "Draws": dr,
    "Minimax_Win_Rate": w1/total if total > 0 else 0,
    "Random_Win_Rate": w2/total if total > 0 else 0,
    "Draw_Rate": dr/total if total > 0 else 0
}
record_results("minimax_vs_random.txt", "Minimax(depth=3) vs Random Agent", results, game_res)

# 3. Minimax vs Q-Learning
# First train Q-learning agent
q_agent = ConnectFourQLearningAgent(alpha=0.1, gamma=0.99, numTraining=10000, epsilon=1)
q_agent.train(numEpisodes=100)  # train fewer episodes for demonstration
# After training, q_agent.epsilon = 0, it acts greedily
# Now play against Minimax
agent_minimax = MinimaxAgent(depth=3)
w1, w2, dr, game_res = run_games(agent_minimax, q_agent, num_games=100)
total = w1 + w2 + dr
results = {
    "Minimax_Depth": 3,
    "Q_TrainingEpisodes": 5000,
    "Wins_MiniMax": w1,
    "Wins_Q": w2,
    "Draws": dr,
    "Minimax_Win_Rate": w1/total if total > 0 else 0,
    "Q_Win_Rate": w2/total if total > 0 else 0,
    "Draw_Rate": dr/total if total > 0 else 0
}
record_results("minimax_vs_qlearning.txt", "Minimax(depth=3) vs Q-Learning(5000 training)", results, game_res)

# 4. Q-Learning vs Q-Learning with various training episodes
training_episodes_list = [1000, 5000, 10000, 20000, 50000]
for t1 in training_episodes_list:
    for t2 in training_episodes_list:
        if t1 == t2:
            continue

        q_agent1 = ConnectFourQLearningAgent(alpha=0.1, gamma=0.99, numTraining=t1, epsilon=1)
        q_agent1.train(numEpisodes=t1)
        q_agent2 = ConnectFourQLearningAgent(alpha=0.1, gamma=0.99, numTraining=t2, epsilon=1)
        q_agent2.train(numEpisodes=t2)

        w1, w2, dr, game_res = run_games(q_agent1, q_agent2, num_games=100)
        total = w1 + w2 + dr
        results = {
            "Q_Agent1_Training": t1,
            "Q_Agent2_Training": t2,
            "Wins_Agent1": w1,
            "Wins_Agent2": w2,
            "Draws": dr,
            "Agent1_Win_Rate": w1/total if total > 0 else 0,
            "Agent2_Win_Rate": w2/total if total > 0 else 0,
            "Draw_Rate": dr/total if total > 0 else 0
        }
        filename = f"qlearning_vs_qlearning_{t1}_{t2}.txt"
        desc = f"Q-Learning vs Q-Learning: Agent1 trained={t1}, Agent2 trained={t2}"
        record_results(filename, desc, results, game_res)

print("All experiments completed. Results are in the data/ directory.")
