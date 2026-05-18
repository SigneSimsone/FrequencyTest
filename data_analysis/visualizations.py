# Visualization Script for Spatial Mapping Experiment
# Creates scatter plots, heatmaps, and spatial visualizations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from data_formatting import load_all_data, calculate_mean_positions


def create_scatter_plot_by_frequency(df, output_path=None):
    
    #Create scatter plots showing all clicks for each frequency (LATVIAN)
    
    frequencies = sorted(df['frequency'].unique())
    n_freq = len(frequencies)

    fig, axes = plt.subplots(1, n_freq, figsize=(5*n_freq, 5))

    if n_freq == 1:
        axes = [axes]

    for idx, freq in enumerate(frequencies):
        freq_data = df[df['frequency'] == freq]

        axes[idx].scatter(freq_data['click_x_norm'],
                         freq_data['click_y_norm'],
                         alpha=0.5, s=50)

        # Add mean position
        mean_x = freq_data['click_x_norm'].mean()
        mean_y = freq_data['click_y_norm'].mean()
        axes[idx].scatter(mean_x, mean_y, color='red', s=200,
                         marker='x', linewidths=3, label='Vidējā pozīcija')

        axes[idx].set_xlim(0, 1)
        axes[idx].set_ylim(0, 1)
        axes[idx].set_xlabel('X pozīcija (normalizēta)', fontsize=11)
        axes[idx].set_ylabel('Y pozīcija (normalizēta)', fontsize=11)
        axes[idx].set_title(f'{freq} Hz', fontsize=12)
        axes[idx].invert_yaxis()  # screen/pixel Y grows downward, so Y=0 is the top of the click area
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()
    return fig


def create_heatmap_by_frequency(df, output_path=None, bins=20):
    
    #Create heatmaps showing density of clicks for each frequency (LATVIAN)
    
    frequencies = sorted(df['frequency'].unique())
    n_freq = len(frequencies)

    fig, axes = plt.subplots(1, n_freq, figsize=(5*n_freq, 5))

    if n_freq == 1:
        axes = [axes]

    for idx, freq in enumerate(frequencies):
        freq_data = df[df['frequency'] == freq]

        # Create 2D histogram
        h, xedges, yedges = np.histogram2d(
            freq_data['click_x_norm'],
            freq_data['click_y_norm'],
            bins=bins,
            range=[[0, 1], [0, 1]]
        )

        # gaussian interpolation adds slight smoothing; change to 'nearest' for a pixel-perfect grid
        im = axes[idx].imshow(h.T, origin='lower', aspect='auto',
                             extent=[0, 1, 0, 1], cmap='YlOrRd',
                             interpolation='gaussian')

        axes[idx].set_xlabel('X pozīcija (normalizēta)', fontsize=11)
        axes[idx].set_ylabel('Y pozīcija (normalizēta)', fontsize=11)
        axes[idx].set_title(f'{freq} Hz - Klikšķu blīvums', fontsize=12)
        axes[idx].invert_yaxis()

        plt.colorbar(im, ax=axes[idx], label='Klikšķu skaits')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()
    return fig


def create_combined_scatter(df, output_path=None):
    
    #Create a single plot with all frequencies overlaid (LATVIAN)
    
    frequencies = sorted(df['frequency'].unique())

    fig, ax = plt.subplots(figsize=(10, 10))

    colors = plt.cm.viridis(np.linspace(0, 1, len(frequencies)))

    for idx, freq in enumerate(frequencies):
        freq_data = df[df['frequency'] == freq]

        ax.scatter(freq_data['click_x_norm'],
                  freq_data['click_y_norm'],
                  alpha=0.4, s=50, color=colors[idx],
                  label=f'{freq} Hz')

        # Add mean position
        mean_x = freq_data['click_x_norm'].mean()
        mean_y = freq_data['click_y_norm'].mean()
        ax.scatter(mean_x, mean_y, color=colors[idx], s=300,
                  marker='x', linewidths=4, edgecolors='black')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('X pozīcija (normalizēta)', fontsize=14)
    ax.set_ylabel('Y pozīcija (normalizēta)', fontsize=14)
    ax.set_title('Klikšķu pozīcijas pēc frekvences', fontsize=16)
    ax.invert_yaxis()
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()
    return fig


def create_position_trends(df, output_path=None):
    
    #Create bar charts showing X and Y positions by frequency (LATVIAN)
    
    summary = calculate_mean_positions(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    frequencies = summary['frequency'].astype(str) + ' Hz'
    x_pos = np.arange(len(frequencies))

    # X position bar chart
    bars1 = ax1.bar(x_pos, summary['mean_x'],
                    yerr=summary['std_x'],
                    capsize=7, alpha=0.7,
                    color='steelblue', edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, summary['mean_x'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(frequencies, fontsize=12)
    ax1.set_xlabel('Frekvence', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Vidējā X pozīcija\n(0=kreisā, 1=labā)', fontsize=13, fontweight='bold')
    ax1.set_title('Horizontālā pozīcija pēc frekvences', fontsize=15, fontweight='bold')
    ax1.set_ylim(0, 1.2)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Vidus')
    ax1.legend()

    # Y position bar chart
    bars2 = ax2.bar(x_pos, summary['mean_y'],
                    yerr=summary['std_y'],
                    capsize=7, alpha=0.7,
                    color='darkorange', edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, summary['mean_y'])):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(frequencies, fontsize=12)
    ax2.set_xlabel('Frekvence', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Vidējā Y pozīcija\n(0=augša, 1=apakša)', fontsize=13, fontweight='bold')
    ax2.set_title('Vertikālā pozīcija pēc frekvences', fontsize=15, fontweight='bold')
    ax2.set_ylim(0, 1.2)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Vidus')
    ax2.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()
    return fig


def create_reaction_time_plot(df, output_path=None):
    
    #Create box plot of reaction times by frequency
    
    fig, ax = plt.subplots(figsize=(10, 6))

    df_plot = df.copy()
    df_plot['frequency'] = df_plot['frequency'].astype(str) + ' Hz'

    sns.boxplot(data=df_plot, x='frequency', y='rt_ms', ax=ax)
    sns.swarmplot(data=df_plot, x='frequency', y='rt_ms',
                 color='black', alpha=0.3, size=3, ax=ax)

    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Reaction Time (ms)', fontsize=12)
    ax.set_title('Reaction Times by Frequency', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()
    return fig


if __name__ == "__main__":
    # Example usage
    data_folder = Path(__file__).parent.parent / "data"
    output_folder = Path(__file__).parent / "figures"
    output_folder.mkdir(exist_ok=True)

    if data_folder.exists():
        print("Loading data...")
        combined_df, participants_df = load_all_data(data_folder)

        print("\nCreating visualizations...")

        # Create all visualizations
        create_scatter_plot_by_frequency(
            combined_df,
            output_folder / "scatter_by_frequency.png"
        )

        create_heatmap_by_frequency(
            combined_df,
            output_folder / "heatmap_by_frequency.png"
        )

        create_combined_scatter(
            combined_df,
            output_folder / "combined_scatter.png"
        )

        create_position_trends(
            combined_df,
            output_folder / "position_trends.png"
        )

        create_reaction_time_plot(
            combined_df,
            output_folder / "reaction_times.png"
        )

        print("\nAll visualizations created!")
    else:
        print(f"Data folder not found: {data_folder}")
