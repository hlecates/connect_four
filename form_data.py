import os
import pandas as pd

# Ensure the output directory exists
formatted_data_dir = os.path.join('data', 'formatted-data')
if not os.path.exists(formatted_data_dir):
    os.makedirs(formatted_data_dir)

summary_rows = []
game_rows = []

# Iterate through all text files in the data directory
for filename in os.listdir('data'):
    if filename.endswith('.txt'):
        filepath = os.path.join('data', filename)
        with open(filepath, 'r') as f:
            lines = f.read().strip().split('\n')

        # Initialize a dictionary to store summary-level results from this file
        summary = {'filename': filename}
        
        # Identify where "Per-game outcomes" section starts, if it exists
        per_game_start = None
        results_section = False
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # When we reach "Results:", subsequent lines up to "Per-game outcomes:" are aggregate data
            if line_stripped.startswith("Results:"):
                results_section = True
                continue
            
            if line_stripped.startswith("Per-game outcomes:"):
                per_game_start = i + 1
                break
            
            # If we are in the results section and the current line has a key-value pair, record it
            if results_section and ':' in line_stripped:
                key, value = line_stripped.split(':', 1)
                key = key.strip()
                value = value.strip()
                summary[key] = value

        # Parse the per-game outcomes if they exist
        if per_game_start is not None:
            for j in range(per_game_start, len(lines)):
                game_line = lines[j].strip()
                if game_line.startswith("Game"):
                    # Format: "Game X: result"
                    prefix, result = game_line.split(':', 1)
                    game_num_str = prefix.replace('Game', '').strip()
                    # Convert game number to int
                    try:
                        game_num = int(game_num_str)
                    except ValueError:
                        # If for some reason parsing fails, skip this line
                        continue
                    result = result.strip()
                    game_rows.append({
                        'filename': filename,
                        'GameNumber': game_num,
                        'Outcome': result
                    })

        summary_rows.append(summary)

# Convert collected data into DataFrames
summary_df = pd.DataFrame(summary_rows)
games_df = pd.DataFrame(game_rows)

# Convert known numeric fields to numeric types
numeric_fields = ['Wins_Agent1', 'Wins_Agent2', 'Draws', 
                  'Agent1_Win_Rate', 'Agent2_Win_Rate', 'Draw_Rate']
for col in numeric_fields:
    if col in summary_df.columns:
        summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')

# Save the data to the new directory
summary_csv_path = os.path.join(formatted_data_dir, 'experiments_summary.csv')
games_csv_path = os.path.join(formatted_data_dir, 'game_outcomes.csv')

summary_df.to_csv(summary_csv_path, index=False)
games_df.to_csv(games_csv_path, index=False)

print(f"Summary results saved to {summary_csv_path}")
print(f"Per-game outcomes saved to {games_csv_path}")
