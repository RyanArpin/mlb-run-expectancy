"""
MLB Run Expectancy Analysis (2023-2025)
=======================================
Calculates run expectancy for all 24 base-out states using Retrosheet data.
Also derives run values per event type and analyzes stolen base / bunt strategy.

Author: Ryan
Data: Retrosheet play-by-play event files (2023-2025)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

print("Loading data...")

df = pd.read_csv(
    '/home/rarpin/baseball_project/data/parsed/events_extended.csv',
    header=None,
    low_memory=False
)

# Rename key columns based on Chadwick field list
cols = {
    0:  'game_id',
    2:  'inning',
    3:  'batting_team',
    4:  'outs',
    26: 'runner_on_1b',
    27: 'runner_on_2b',
    28: 'runner_on_3b',
    29: 'event_text',
    34: 'event_type',
}
df = df.rename(columns=cols)

# Calculate runs scored from batter/runner destination columns
# Destination >= 4 means the player scored
df['runs_on_play'] = sum((df[col] >= 4).astype(int) for col in [58, 59, 60, 61])

print(f"Loaded {len(df):,} plays.")

# ─────────────────────────────────────────────
# 2. CREATE BASE STATES
# ─────────────────────────────────────────────

# Each base is 1 if occupied, 0 if empty
# Base state is a 3-character string: "1B-2B-3B" e.g. "100" = runner on 1st only
df['on_1b'] = df['runner_on_1b'].notna().astype(int)
df['on_2b'] = df['runner_on_2b'].notna().astype(int)
df['on_3b'] = df['runner_on_3b'].notna().astype(int)
df['base_state'] = df['on_1b'].astype(str) + df['on_2b'].astype(str) + df['on_3b'].astype(str)

# ─────────────────────────────────────────────
# 3. RUNS TO END OF INNING
# ─────────────────────────────────────────────

# Create a unique identifier for each half-inning
df['half_inning'] = df['game_id'] + '_' + df['inning'].astype(str) + '_' + df['batting_team'].astype(str)

# Reverse cumulative sum gives runs scored from this play to end of inning
df['runs_to_end'] = df.groupby('half_inning')['runs_on_play'].transform(
    lambda x: x[::-1].cumsum()[::-1]
)

# ─────────────────────────────────────────────
# 4. RUN EXPECTANCY MATRIX
# ─────────────────────────────────────────────

# Filter to valid in-play states (0, 1, 2 outs only)
df_valid = df[df['outs'] < 3].copy()

# Average runs to end of inning for each of the 24 base-out states
run_expectancy = df_valid.groupby(['outs', 'base_state'])['runs_to_end'].mean().reset_index()
run_expectancy.columns = ['outs', 'base_state', 'run_expectancy']

# Pivot into 8x3 matrix
matrix = run_expectancy.pivot(index='base_state', columns='outs', values='run_expectancy')
matrix.columns = ['0 outs', '1 out', '2 outs']

# Sort rows by average run expectancy ascending
state_order = matrix.mean(axis=1).sort_values().index
matrix = matrix.loc[state_order]

print("\nRun Expectancy Matrix:")
print(matrix.round(3))

# Build lookup dictionary for run expectancy by (outs, base_state)
re_dict = {}
for _, row in run_expectancy.iterrows():
    re_dict[(int(row['outs']), row['base_state'])] = row['run_expectancy']

# ─────────────────────────────────────────────
# 5. RUN VALUES PER EVENT TYPE
# ─────────────────────────────────────────────

# Map numeric base state after play (extended field 14, column 111)
# Encoded as binary: 0=000, 1=100, 2=010, 3=110, 4=001, 5=101, 6=011, 7=111
base_state_map = {
    0: '000', 1: '100', 2: '010', 3: '110',
    4: '001', 5: '101', 6: '011', 7: '111'
}
df_valid['base_state_after'] = df[111].map(base_state_map)
df_valid['outs_after'] = df_valid['outs'] + df_valid[40]

# Run expectancy before and after each play
df_valid['re_before'] = df_valid.apply(
    lambda r: re_dict.get((r['outs'], r['base_state']), 0), axis=1
)
df_valid['re_after'] = df_valid.apply(
    lambda r: re_dict.get((int(r['outs_after']), r['base_state_after']), 0)
    if r['outs_after'] < 3 else 0, axis=1
)

# Run value = change in run expectancy + runs actually scored
df_valid['run_value'] = df_valid['re_after'] - df_valid['re_before'] + df_valid['runs_on_play']

# Map event type codes to readable names (verified against Retrosheet event text)
event_types = {
    2:  'Field Out',
    3:  'Strikeout',
    4:  'Stolen Base',
    5:  'Def. Indifference',
    6:  'Caught Stealing',
    9:  'Wild Pitch',
    14: 'Walk',
    15: 'Int. Walk',
    16: 'Hit by Pitch',
    18: 'Error',
    19: "Fielder's Choice",
    20: 'Single',
    21: 'Double',
    22: 'Triple',
    23: 'Home Run',
}
df_valid['event_name'] = df_valid['event_type'].map(event_types)

event_values = df_valid.groupby('event_name')['run_value'].mean().sort_values(ascending=False)

print("\nRun Value by Event Type:")
print(event_values.round(3))

# ─────────────────────────────────────────────
# 6. STRATEGIC ANALYSIS
# ─────────────────────────────────────────────

print("\nStolen Base 2nd - Break-even Success Rate")
print("-" * 45)
sb2_rates = []
for outs in [0, 1, 2]:
    re_before  = re_dict[(outs, '100')]
    re_success = re_dict[(outs, '010')]
    re_fail    = re_dict[(min(outs + 1, 2), '000')] if outs < 2 else 0
    break_even = (re_before - re_fail) / (re_success - re_fail)
    sb2_rates.append(break_even)
    print(f"  {outs} outs: must succeed {break_even:.1%} of the time")

print("\nStolen Base 3rd - Break-even Success Rate")
print("-" * 45)
sb3_rates = []
for outs in [0, 1, 2]:
    re_before  = re_dict[(outs, '010')]
    re_success = re_dict[(outs, '001')]
    re_fail    = re_dict[(min(outs + 1, 2), '000')] if outs < 2 else 0
    break_even = (re_before - re_fail) / (re_success - re_fail)
    sb3_rates.append(break_even)
    print(f"  {outs} outs: must succeed {break_even:.1%} of the time")

print("\nSacrifice Bunt - Run Expectancy Change")
print("-" * 45)
bunt_situations = [
    ('100', '010', 0, 1, 'Runner on 1st, 0 outs'),
    ('010', '001', 0, 1, 'Runner on 2nd, 0 outs'),
    ('110', '011', 0, 1, 'Runners on 1st+2nd, 0 outs'),
]
bunt_changes = []
for base_before, base_after, outs_before, outs_after, label in bunt_situations:
    re_before = re_dict[(outs_before, base_before)]
    re_after  = re_dict[(outs_after,  base_after)]
    change    = re_after - re_before
    bunt_changes.append(change)
    verdict   = "WORTH IT" if change > 0 else "NOT WORTH IT"
    print(f"  {label}: {re_before:.3f} -> {re_after:.3f} ({change:+.3f}) — {verdict}")

# ─────────────────────────────────────────────
# 7. VISUALIZATIONS
# ─────────────────────────────────────────────

print("\nGenerating plots...")

# --- Plot 1: Run Expectancy Heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd',
            linewidths=0.5, cbar_kws={'label': 'Expected Runs'})
plt.title('MLB Run Expectancy Matrix (2023-2025)', fontsize=14)
plt.xlabel('Outs')
plt.ylabel('Base State (1B-2B-3B)')
plt.tight_layout()
plt.savefig('/home/rarpin/baseball_project/notebooks/run_expectancy_matrix.png', dpi=150)
plt.show()

# --- Plot 2: Run Value by Event Type ---
ev_plot = event_values.dropna()
plt.figure(figsize=(10, 6))
colors = ['crimson' if v > 0 else 'steelblue' for v in ev_plot.values]
plt.barh(ev_plot.index, ev_plot.values, color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Run Value by Event Type (2023-2025)', fontsize=14)
plt.xlabel('Average Run Value')
plt.tight_layout()
plt.savefig('/home/rarpin/baseball_project/notebooks/run_values.png', dpi=150)
plt.show()

# --- Plot 3: Strategic Analysis ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

situations = ['0 outs', '1 out', '2 outs']
x = np.arange(3)
ax1.bar(x - 0.2, sb2_rates, 0.4, label='Steal 2nd', color='crimson')
ax1.bar(x + 0.2, sb3_rates, 0.4, label='Steal 3rd', color='steelblue')
ax1.set_xticks(x)
ax1.set_xticklabels(situations)
ax1.set_ylabel('Break-even Success Rate')
ax1.set_title('Stolen Base Break-even Rates')
ax1.set_ylim(0, 1)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax1.legend()

bunt_labels = ['Runner on 1st', 'Runner on 2nd', '1st and 2nd']
ax2.bar(bunt_labels, bunt_changes, color='steelblue')
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_ylabel('Change in Run Expectancy')
ax2.set_title('Sacrifice Bunt — Run Expectancy Impact')
ax2.set_ylim(-0.3, 0.1)

plt.suptitle('MLB Strategic Analysis (2023-2025)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/rarpin/baseball_project/notebooks/strategic_analysis.png', dpi=150)
plt.show()

print("\nDone! All plots saved to ~/baseball_project/notebooks/")
