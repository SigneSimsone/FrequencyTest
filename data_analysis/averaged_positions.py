# Averaged Position Visualizations
# Shows one averaged point per participant per frequency
# (averaging across the 3 repetitions)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from data_formatting import load_all_data


def calculate_averaged_positions(df):
    
    #Calculate the average position for each participant for each frequency
    #(averaging across the 3 trials)
    
    averaged = df.groupby(['participant_id', 'frequency']).agg({
        'click_x_norm': 'mean',
        'click_y_norm': 'mean',
        'rt_ms': 'mean'
    }).reset_index()

    averaged.columns = ['participant_id', 'frequency',
                       'avg_x', 'avg_y', 'avg_rt']

    return averaged


def plot_averaged_scatter_by_frequency(df, output_path=None):
    
    #Create scatter plots showing ONE averaged point per participant per frequency (LATVIAN)
    
    averaged_df = calculate_averaged_positions(df)
    frequencies = sorted(averaged_df['frequency'].unique())
    n_freq = len(frequencies)

    fig, axes = plt.subplots(1, n_freq, figsize=(5*n_freq, 5))

    if n_freq == 1:
        axes = [axes]

    for idx, freq in enumerate(frequencies):
        freq_data = averaged_df[averaged_df['frequency'] == freq]

        # Plot each participant's averaged position
        axes[idx].scatter(freq_data['avg_x'],
                         freq_data['avg_y'],
                         alpha=0.6, s=100,
                         color='steelblue',
                         edgecolors='black',
                         linewidths=1)

        # Add overall mean position (mean of means)
        mean_x = freq_data['avg_x'].mean()
        mean_y = freq_data['avg_y'].mean()
        axes[idx].scatter(mean_x, mean_y, color='red', s=300,
                         marker='X', linewidths=3,
                         edgecolors='black',
                         label='Kopējais vidējais', zorder=10)

        axes[idx].set_xlim(0, 1)
        axes[idx].set_ylim(0, 1)
        axes[idx].set_xlabel('X pozīcija (normalizēta)', fontsize=11)
        axes[idx].set_ylabel('Y pozīcija (normalizēta)', fontsize=11)
        axes[idx].set_title(f'{freq} Hz\n(n={len(freq_data)} dalībnieki)',
                           fontsize=12)
        axes[idx].invert_yaxis()
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Vidējās pozīcijas katram dalībniekam\n(Katrs punkts = 3 mēģinājumu vidējais)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()
    return fig


def plot_averaged_combined_scatter(df, output_path=None):
    
    #Create a single plot with averaged positions for all frequencies overlaid (LATVIAN)
    
    averaged_df = calculate_averaged_positions(df)
    frequencies = sorted(averaged_df['frequency'].unique())

    fig, ax = plt.subplots(figsize=(10, 10))

    colors = plt.cm.viridis(np.linspace(0, 1, len(frequencies)))

    for idx, freq in enumerate(frequencies):
        freq_data = averaged_df[averaged_df['frequency'] == freq]

        # Plot averaged positions
        ax.scatter(freq_data['avg_x'],
                  freq_data['avg_y'],
                  alpha=0.5, s=100, color=colors[idx],
                  label=f'{freq} Hz (n={len(freq_data)})',
                  edgecolors='black', linewidths=0.5)

        # Add overall mean position
        mean_x = freq_data['avg_x'].mean()
        mean_y = freq_data['avg_y'].mean()
        ax.scatter(mean_x, mean_y, color=colors[idx], s=400,
                  marker='X', linewidths=4, edgecolors='black', zorder=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('X pozīcija (normalizēta)', fontsize=14)
    ax.set_ylabel('Y pozīcija (normalizēta)', fontsize=14)
    ax.set_title('Vidējās klikšķu pozīcijas pēc frekvences\n(Katrs punkts = viens dalībnieks, vidējais no 3 mēģinājumiem)',
                 fontsize=16)
    ax.invert_yaxis()
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()
    return fig


def plot_sharp_heatmap_by_frequency(df, output_path=None, bins=20):
    
    #Create heatmaps showing density of averaged clicks with gaussian blur (LATVIAN)
    
    averaged_df = calculate_averaged_positions(df)
    frequencies = sorted(averaged_df['frequency'].unique())
    n_freq = len(frequencies)

    fig, axes = plt.subplots(1, n_freq, figsize=(5*n_freq, 5))

    if n_freq == 1:
        axes = [axes]

    for idx, freq in enumerate(frequencies):
        freq_data = averaged_df[averaged_df['frequency'] == freq]

        # Create 2D histogram
        h, xedges, yedges = np.histogram2d(
            freq_data['avg_x'],
            freq_data['avg_y'],
            bins=bins,
            range=[[0, 1], [0, 1]]
        )

        # Plot heatmap with gaussian blur
        im = axes[idx].imshow(h.T, origin='lower', aspect='auto',
                             extent=[0, 1, 0, 1], cmap='YlOrRd',
                             interpolation='gaussian')

        axes[idx].set_xlabel('X pozīcija (normalizēta)', fontsize=11)
        axes[idx].set_ylabel('Y pozīcija (normalizēta)', fontsize=11)
        axes[idx].set_title(f'{freq} Hz - Blīvums\n(n={len(freq_data)} dalībnieki)',
                           fontsize=12)
        axes[idx].invert_yaxis()

        plt.colorbar(im, ax=axes[idx], label='Dalībnieku skaits')

    plt.suptitle('Pozīciju blīvuma siltumkarte (Vidējais katram dalībniekam)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()
    return fig


def plot_variance_by_participant(df, output_path=None):
    
    #Show how much variance each participant has in their responses
    #(how consistent they are across the 3 trials)
    
    variance_data = []

    for participant in df['participant_id'].unique():
        p_data = df[df['participant_id'] == participant]

        for freq in p_data['frequency'].unique():
            freq_data = p_data[p_data['frequency'] == freq]

            if len(freq_data) >= 2:
                std_x = freq_data['click_x_norm'].std()
                std_y = freq_data['click_y_norm'].std()
                spatial_std = np.sqrt(std_x**2 + std_y**2)

                variance_data.append({
                    'participant_id': participant,
                    'frequency': freq,
                    'spatial_std': spatial_std
                })

    variance_df = pd.DataFrame(variance_data)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(data=variance_df, x='frequency', y='spatial_std', ax=ax)
    sns.swarmplot(data=variance_df, x='frequency', y='spatial_std',
                 color='black', alpha=0.3, size=4, ax=ax)

    ax.set_xlabel('Frekvence (Hz)', fontsize=12)
    ax.set_ylabel('Telpiskā mainība (std novirze)', fontsize=12)
    ax.set_title('Dalībnieku konsekvence\n(Zemāks = konsekventāks 3 mēģinājumos)',
                 fontsize=14)
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
    output_folder = Path(__file__).parent / "output"
    figures_folder = output_folder / "figures"
    results_folder = output_folder / "results"

    figures_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)

    if data_folder.exists():
        print("Loading data...")
        combined_df, participants_df = load_all_data(data_folder)

        print("\nCreating averaged position visualizations...")

        # Averaged scatter plots
        plot_averaged_scatter_by_frequency(
            combined_df,
            figures_folder / "averaged_scatter_by_frequency.png"
        )

        plot_averaged_combined_scatter(
            combined_df,
            figures_folder / "averaged_combined_scatter.png"
        )

        # Sharp heatmap
        plot_sharp_heatmap_by_frequency(
            combined_df,
            figures_folder / "sharp_heatmap_by_frequency.png"
        )

        # Consistency visualization
        plot_variance_by_participant(
            combined_df,
            figures_folder / "participant_consistency_plot.png"
        )

        # Save averaged data
        averaged_df = calculate_averaged_positions(combined_df)
        averaged_df.to_csv(results_folder / "averaged_positions.csv", index=False)

        print("\nAveraged position analysis complete!")
    else:
        print(f"Data folder not found: {data_folder}")
