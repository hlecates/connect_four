import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
