# Visualizing Crowd Motion Prediction Datasets
# Feature columns: frame_id, pedestrian_id, x, y

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
PLOT_DIR = os.path.dirname(__file__)

# List all data files
files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]

for fname in files:
    path = os.path.join(DATA_DIR, fname)
    # Read the data
    df = pd.read_csv(path, delim_whitespace=True, header=None, names=['frame_id', 'pedestrian_id', 'x', 'y'])
    print(f"Loaded {fname}: {df.shape[0]} rows, {df['pedestrian_id'].nunique()} pedestrians, {df['frame_id'].nunique()} frames")

    # Plot all trajectories
    plt.figure(figsize=(8, 6))
    for pid, group in df.groupby('pedestrian_id'):
        plt.plot(group['x'], group['y'], marker='o', markersize=2, linewidth=1, label=f'Ped {pid}' if pid < 5 else None)
    plt.title(f"Trajectories in {fname}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='best', fontsize='small', ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{fname}_trajectories.png"))
    plt.close()

    # Plot heatmap of all positions
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=df['x'], y=df['y'], fill=True, cmap='viridis', bw_adjust=0.5)
    plt.title(f"Position Density in {fname}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{fname}_heatmap.png"))
    plt.close()

    # Plot number of pedestrians per frame
    plt.figure(figsize=(8, 4))
    frame_counts = df.groupby('frame_id')['pedestrian_id'].nunique()
    plt.plot(frame_counts.index, frame_counts.values)
    plt.title(f"Pedestrians per Frame in {fname}")
    plt.xlabel('Frame ID')
    plt.ylabel('Number of Pedestrians')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{fname}_pedestrians_per_frame.png"))
    plt.close()

print("Plots saved in:", PLOT_DIR)
