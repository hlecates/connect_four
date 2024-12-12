# Connect Four AI and Experimentation

This repository provides an implementation of the classic Connect Four game along with several AI agents, including a Minimax-based AI and a Q-Learning-based reinforcement learning agent. It also includes a GUI for interactive play and code for running experiments to compare different agents and configurations.

## Features

- **Human vs. Human**: Play against another human on the same machine.
- **Human vs. Minimax Agent**: Challenge a Minimax-based AI with a configurable search depth.
- **Human vs. Q-Learning Agent**: Play against a Q-Learning-trained agent that learns from experience.
- **Automated Experiments**: Run batches of games between different agents (Minimax, Q-Learning, Random) to gather statistics and assess performance.
- **Configurable Parameters**: Adjust Minimax search depth, Q-Learning parameters (alpha, gamma, epsilon), training duration, and more.
- **GUI Interface**: A simple Tkinter-based GUI to visualize the board, interact via mouse clicks, and view real-time results.

## File Overview

- **`connectFour.py`**:
  - Parses command-line arguments:
    - No arguments: Start Human vs. Human.
    - `minimax`: Start Human vs. Minimax agent.
    - `qlearning`: Start Human vs. Q-Learning agent (after training).
  - Creates a game board and initializes the GUI.
  - Depending on the mode, sets up the appropriate agent and starts the game loop.

- **agents/**:
  - **`minimaxAgent.py`**: Implements the Minimax algorithm with alpha-beta pruning. Allows configuring search depth.
  - **`qlearningAgent.py`**: Implements a Q-Learning agent that trains by interacting with the environment, storing Q-values, and updating them based on rewards.
  - **`randomAgent.py`**: A simple opponent agent that picks moves uniformly at random.

- **`gui.py`**:
  - **`ConnectFourGUI`**: A Tkinter-based graphical interface for Connect Four.  
    Displays the board and handles human interactions.

- **`utils.py`**:
  - Contains helper functions for creating boards, validating moves, encoding/decoding states, detecting terminal states, and placing pieces.

- **Experimentation code** (included at the bottom of the main file and some commented blocks):
  - Demonstrates how to run simulations between various agents (e.g., Minimax vs Random, Q-Learning vs Minimax) and record results in the `data/` directory.
  - `run_games()` and `record_results()` functions facilitate batch testing and result logging.


## Running the Game

1. **Player vs Player**:
   ```bash
   python connectFour.py
3. **Minimax vs Player**:
   ```bash
   python connectFour.py minimax
5. **Q-Learning vs Player**:
   ```bash
   python connectFour.py qlearning
