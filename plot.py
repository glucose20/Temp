import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PLOT_FOLDER = "plots"
TARGET_METRICS = ['mse_mean', 'rmse_mean', 'pearson_mean', 'spearman_mean', 'ci_mean', 'r2_mean', 'final_entropy_mean', 'ci_std', 'mse_std']

# Create plots directory if it doesn't exist
Path(PLOT_FOLDER).mkdir(parents=True, exist_ok=True)

# Load the aggregated summary stats with 3 header levels
summary_stats = pd.read_csv("sweep_all_summary_stats_m.csv", header=[0, 1, 2])

# Flatten MultiIndex columns
summary_stats.columns = ['_'.join([col_name if "Unnamed" not in col_name else '' for col_name in col]).strip('_') for col in summary_stats.columns.values]

print(summary_stats.columns)

# Map running_set to marker shapes for consistent styling
marker_shapes = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'H']
running_set_markers = {rs: marker_shapes[i % len(marker_shapes)]
                       for i, rs in enumerate(summary_stats['running_set'].unique())}

# Map configs to numeric values for colors
config_mapping = {config: i for i, config in enumerate(summary_stats['config'].unique())}
summary_stats['config_numeric'] = summary_stats['config'].map(config_mapping)

combos = summary_stats[['dataset', 'running_set']].drop_duplicates().reset_index(drop=True)
num_plots = len(combos)
ncols = min(4, num_plots) if num_plots else 1
nrows = int(np.ceil(num_plots / ncols)) if num_plots else 1

for metric in TARGET_METRICS:
    # Create subplots for each dataset + running_set combination
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
    axes = np.atleast_1d(axes).flatten()

    lr_min = summary_stats['learning_rate'].str[2:].astype(float).min()
    lr_max = summary_stats['learning_rate'].str[2:].astype(float).max()

    for idx, combo in combos.iterrows():
        ax = axes[idx]
        dataset = combo['dataset']
        running_set = combo['running_set']
        subset = summary_stats[(summary_stats['dataset'] == dataset) & (summary_stats['running_set'] == running_set)]

        scatter = ax.scatter(
            subset['config_numeric'],
            subset[metric],
            s=[int(bs[1:])/5 for bs in subset['batch_size']],  # Scale batch size for visibility
            c=[float(lr[2:]) for lr in subset['learning_rate']],  # Extract learning rate value
            cmap='viridis',
            vmin=lr_min,
            vmax=lr_max,
            marker=running_set_markers.get(running_set, 'o'),
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
            label=f'Running Set: {running_set}'
        )

        # Draw lines connecting points with the same learning rate and batch size within this running_set
        for lr in subset['learning_rate'].unique():
            for batch_size in subset['batch_size'].unique():
                lr_data = subset[(subset['learning_rate'] == lr) & (subset['batch_size'] == batch_size)].sort_values('config_numeric')
                if len(lr_data) > 1:
                    lr_value = float(lr[2:])
                    normalized_lr = (lr_value - lr_min) / (lr_max - lr_min) if lr_max > lr_min else 0.5
                    color = plt.cm.viridis(normalized_lr)
                    bs_value = int(batch_size[1:]) / 100  # Scale batch size for visibility
                    ax.plot(
                        lr_data['config_numeric'],
                        lr_data[metric],
                        color=color,
                        alpha=0.6,
                        linewidth=bs_value,
                        zorder=0
                    )

        ax.set_ylabel(f'{str(metric).capitalize()} Mean', fontsize=11, fontweight='bold')
        ax.set_title(f'Dataset: {dataset} | Running Set: {running_set}', fontsize=12, fontweight='bold')
        ax.set_xticks(list(config_mapping.values()))
        ax.set_xticklabels([e[4:] if 'moe' in e else e for e in list(config_mapping.keys())], rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')

    # Hide any unused subplots (if num_plots is not a perfect grid)
    for ax in axes[num_plots:]:
        ax.axis('off')


    # Create a common colorbar for learning rate
    # Increase horizontal/vertical spacing to avoid axes overlap
    fig.subplots_adjust(right=0.85, hspace=0.35, wspace=0.25, top=0.9)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                                norm=plt.Normalize(vmin=summary_stats['learning_rate'].str[2:].astype(float).min(),
                                                vmax=summary_stats['learning_rate'].str[2:].astype(float).max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, label='Learning Rate')

    fig.suptitle(f'{str(metric).capitalize()} vs Config by Dataset and Running Set (Size: Batch Size, Color: Learning Rate)', 
                fontsize=14, fontweight='bold', y=0.95)


    # create subfolder for metric if it doesn't exist
    metric_folder = f"{PLOT_FOLDER}/{metric}"
    Path(metric_folder).mkdir(parents=True, exist_ok=True)

    # Save figure
    plt.savefig(f'{metric_folder}/{metric}_vs_batch_size.png', dpi=300, bbox_inches='tight')
    print(f"Chart saved as '{metric}_vs_batch_size.png'")

    # Display the plot
    plt.show()
