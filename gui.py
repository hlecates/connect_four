import tkinter as tk
import random
from tkinter import messagebox
from utils import is_terminal_board, detect_win, place_piece, HUMAN, AI, valid_locations, ROWS, COLUMNS, create_board, encode_state

class ConnectFourGUI:
    def __init__(self, board):
        self.board = board
        self.cell_size = 100
        self.width = COLUMNS * self.cell_size
        self.height = ROWS * self.cell_size
        self.root = tk.Tk()
        self.root.title("Connect Four")
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg='yellow')
        self.canvas.pack()

        # Bind mouse click event
        self.canvas.bind("<Button-1>", self.handle_click)

        # Game state variables
        self.current_player = HUMAN
        self.human_vs_human = True
        self.ai_agent = None
        self.game_over = False

        self.draw_board()

    def run_game(self, human_vs_human=True, ai_agent=None):
        self.human_vs_human = human_vs_human
        self.ai_agent = ai_agent
        self.current_player = HUMAN
        self.game_over = False
        self.draw_board()
        self.root.mainloop()

    def draw_board(self):
        # Clear the canvas
        self.canvas.delete("all")

        # Draw the slots
        for row in range(ROWS):
            for col in range(COLUMNS):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                piece = self.board[row][col]
                if piece == HUMAN:
                    color = 'blue'
                elif piece == AI:
                    color = 'red'
                else:
                    color = 'white'
                self.canvas.create_oval(x1+5, y1+5, x2-5, y2-5, fill=color, outline='black', width=2)

    def handle_click(self, event):
        if self.game_over:
            return

        col = event.x // self.cell_size

        # If invalid move, ignore
        if col not in valid_locations(self.board):
            return

        # Human move
        place_piece(self.board, self.current_player, col)
        self.draw_board()
        
        # Now that the move is placed and shown, check for game end
        if is_terminal_board(self.board):
            self.end_game()
            return

        # Switch player or invoke AI move if playing vs AI
        if self.human_vs_human:
            self.current_player = AI if self.current_player == HUMAN else HUMAN
        else:
            # Schedule AI move after a short delay to ensure the board has been updated visually
            self.root.after(300, self.ai_move)

    def ai_move(self):
        if self.game_over or not self.ai_agent:
            return

        valid_moves = valid_locations(self.board)
        if len(valid_moves) > 0:
            state = encode_state(self.board)
            ai_col = self.ai_agent.getAction(self.board)
            # Ensure AI's move is valid
            if ai_col not in valid_moves:
                ai_col = random.choice(valid_moves)

            place_piece(self.board, AI, ai_col)
            self.draw_board()

            # After placing AI's piece and updating the display, check terminal condition
            if is_terminal_board(self.board):
                self.end_game()
                return

            # Switch back to human
            self.current_player = HUMAN

    def end_game(self):
        self.game_over = True
        # Determine result
        if detect_win(self.board, HUMAN):
            winner = "HUMAN (Blue)"
        elif detect_win(self.board, AI):
            winner = "AI (Red)" if not self.human_vs_human else "HUMAN 2 (Red)"
        else:
            winner = "No one! It's a draw."

        # Show winner message before destroying the window
        messagebox.showinfo("Game Over", f"Game Over! Winner: {winner}")
        self.root.destroy()
