
import os
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt


# Load the summary data
df = pd.read_csv('data/formatted-data/experiments_summary.csv')

# Ensure the necessary fields are numeric
df['Q_Agent1_Training'] = pd.to_numeric(df['Q_Agent1_Training'], errors='coerce')
df['Q_Agent2_Training'] = pd.to_numeric(df['Q_Agent2_Training'], errors='coerce')
df['Agent1_Win_Rate'] = pd.to_numeric(df['Agent1_Win_Rate'], errors='coerce')
df['Agent2_Win_Rate'] = pd.to_numeric(df['Agent2_Win_Rate'], errors='coerce')

# Filter to Q-Learning vs Q-Learning experiments
q_vs_q = df.dropna(subset=['Q_Agent1_Training', 'Q_Agent2_Training'])

# Fix Agent1 at 1000 training episodes
subset = q_vs_q[q_vs_q['Q_Agent1_Training'] == 1000].copy()

# Sort by Agent2 training episodes
subset.sort_values('Q_Agent2_Training', inplace=True)

# Plot Agent2's win rate vs. Agent2's training episodes
plt.figure(figsize=(8,6))
sns.lineplot(x='Q_Agent2_Training', y='Agent2_Win_Rate', data=subset, marker='o')

plt.title('Q-Learning Agent2 Win Rate vs Training Episodes (Agent1 fixed at 1000)')
plt.xlabel('Agent2 Training Episodes')
plt.ylabel('Agent2 Win Rate')
plt.grid(True)
plt.show()



# Ensure the images directory exists
images_dir = os.path.join('data', 'images')
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Load the summary data
df = pd.read_csv('data/formatted-data/experiments_summary.csv')

# Filter to only minimax vs minimax experiments by checking depth columns
minimax_df = df.dropna(subset=['Minimax_Depth_Agent1', 'Minimax_Depth_Agent2'])

# Include Draw_Rate as well
plot_data = minimax_df[['filename', 'Agent1_Win_Rate', 'Agent2_Win_Rate', 'Draw_Rate']].copy()

# Convert to numeric if needed
for col in ['Agent1_Win_Rate', 'Agent2_Win_Rate', 'Draw_Rate']:
    plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')

# Iterate through each filename and plot separately
for filename, group in plot_data.groupby('filename'):
    # Melt the group to have a column indicating Agent1, Agent2, or Draw
    melted = pd.melt(group, 
                     id_vars='filename', 
                     value_vars=['Agent1_Win_Rate', 'Agent2_Win_Rate', 'Draw_Rate'],
                     var_name='ResultType', 
                     value_name='Rate')
    
    # Rename the ResultType values to more readable labels
    # Agent1_Win_Rate -> Agent1, Agent2_Win_Rate -> Agent2, Draw_Rate -> Draw
    melted['ResultType'] = melted['ResultType'].replace({
        'Agent1_Win_Rate': 'Agent1',
        'Agent2_Win_Rate': 'Agent2',
        'Draw_Rate': 'Draw'
    })
    
    plt.figure(figsize=(6,4))
    sns.barplot(x='ResultType', y='Rate', data=melted)
    plt.title(f'Win/Draw Rates for {filename}')
    plt.xlabel('Outcome')
    plt.ylabel('Rate')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    # Save the figure as PNG in the images directory
    safe_filename = filename.replace('.txt', '')
    out_path = os.path.join(images_dir, f"{safe_filename}_rates.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

raw_data_dir = 'data/raw'

# Initialize a list to hold summary results
summary_rows = []

# Iterate over the raw data files
for filename in os.listdir(raw_data_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(raw_data_dir, filename)
        
        # Parse the file
        with open(filepath, 'r') as f:
            lines = f.read().strip().split('\n')

        # We'll look for the "Results:" section and parse key-value pairs after it
        results_section = False
        summary = {'filename': filename}
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("Results:"):
                results_section = True
                continue
            if results_section and ':' in line_stripped:
                # Extract key-value
                key, value = line_stripped.split(':', 1)
                key = key.strip()
                value = value.strip()

                # Store in summary
                summary[key] = value

            # Stop when we reach "Per-game outcomes" if that occurs
            if line_stripped.startswith("Per-game outcomes:"):
                # We've extracted all summary info we need
                break
        
        # Append summary to the list
        summary_rows.append(summary)

# Convert to DataFrame
summary_df = pd.DataFrame(summary_rows)

# Convert numeric fields
if 'Q_TrainingEpisodes' in summary_df.columns:
    summary_df['Q_TrainingEpisodes'] = pd.to_numeric(summary_df['Q_TrainingEpisodes'], errors='coerce')
if 'Q_Win_Rate' in summary_df.columns:
    summary_df['Q_Win_Rate'] = pd.to_numeric(summary_df['Q_Win_Rate'], errors='coerce')

# Filter out rows that don't have Q_TrainingEpisodes or Q_Win_Rate
summary_df = summary_df.dropna(subset=['Q_TrainingEpisodes', 'Q_Win_Rate'])

# Sort by Q_TrainingEpisodes for a cleaner plot
summary_df.sort_values('Q_TrainingEpisodes', inplace=True)

# Plot the training episodes vs win rate
plt.figure(figsize=(8,6))
sns.lineplot(x='Q_TrainingEpisodes', y='Q_Win_Rate', data=summary_df, marker='o')
plt.title("Q-Learning Agent Win Rate vs Training Episodes (vs Random)")
plt.xlabel("Training Episodes")
plt.ylabel("Win Rate")
plt.grid(True)
plt.tight_layout()
plt.show()
