import sys
from utils import create_board, is_terminal_board, detect_win, valid_locations
from gui import ConnectFourGUI 
from agents.randomAgent import RandomAgent
from agents.minimaxAgent import MinimaxAgent
from agents.qlearningAgent import ConnectFourQLearningAgent

def main():
    args = sys.argv[1:]
    mode = None
    if len(args) == 0:
        mode = 'human'
    else:
        mode = args[0].lower()

    board = create_board()
    
    # Initialize GUI or terminal interface
    # For demonstration, assume we have a GUI class that manages the game loop:
    # The GUI class will handle rendering and getting human moves.
    gui = ConnectFourGUI(board)

    if mode == 'human':
        # Two human players
        gui.run_game(human_vs_human=True)
    elif mode == 'minimax':
        # Human vs minimax
        agent = MinimaxAgent(depth=5)
        gui.run_game(ai_agent=agent, human_vs_human=False)
    elif mode == 'qlearning':
        q_agent = ConnectFourQLearningAgent(alpha=0.05, gamma=0.8, numTraining=10000, epsilon=1)
        #opponent = RandomAgent()
        opponent = MinimaxAgent(depth=1)
        q_agent.train()
        gui.run_game(ai_agent=q_agent, human_vs_human=False)

    else:
        print("Unknown mode. Use 'minimax' or 'qlearning', or no argument for human vs human.")

if __name__ == "__main__":
    main()
