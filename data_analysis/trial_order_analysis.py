# Trial Order Analysis for Spatial Mapping Experiment
# Analyzes whether spatial mappings change across repeated exposures
# (1st, 2nd, 3rd time hearing the same frequency)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from data_formatting import load_all_data


def add_trial_order(df):
    
    #Add a column indicating which occurrence this is (1st, 2nd, 3rd time)
    #for each frequency per participant
    
    df = df.copy()

    # cumcount() is 0-indexed, +1 makes trial_order 1, 2, 3 (first, second, third hearing)
    df['trial_order'] = df.groupby(['participant_id', 'frequency']).cumcount() + 1

    return df


def calculate_position_by_trial_order(df):
    
    #Calculate mean positions for each frequency at each trial order (1st, 2nd, 3rd)
    
    df = add_trial_order(df)

    summary = df.groupby(['frequency', 'trial_order']).agg({
        'click_x_norm': ['mean', 'std', 'count'],
        'click_y_norm': ['mean', 'std'],
        'rt_ms': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    summary.columns = ['frequency', 'trial_order',
                       'mean_x', 'std_x', 'count',
                       'mean_y', 'std_y',
                       'mean_rt', 'std_rt']

    return summary


def test_trial_order_effect(df):
    
    #Test if there are significant differences in positions across trial orders
    #using repeated measures ANOVA or Friedman test
    
    df = add_trial_order(df)

    results = {}

    for freq in sorted(df['frequency'].unique()):
        freq_data = df[df['frequency'] == freq]

        # Get data for each trial order
        order_1 = freq_data[freq_data['trial_order'] == 1]
        order_2 = freq_data[freq_data['trial_order'] == 2]
        order_3 = freq_data[freq_data['trial_order'] == 3]

        # Skip if we don't have all three orders
        if len(order_1) == 0 or len(order_2) == 0 or len(order_3) == 0:
            continue

        # Test for X position
        f_stat_x, p_value_x = stats.f_oneway(
            order_1['click_x_norm'],
            order_2['click_x_norm'],
            order_3['click_x_norm']
        )

        # Test for Y position
        f_stat_y, p_value_y = stats.f_oneway(
            order_1['click_y_norm'],
            order_2['click_y_norm'],
            order_3['click_y_norm']
        )

        # Test for reaction time
        f_stat_rt, p_value_rt = stats.f_oneway(
            order_1['rt_ms'],
            order_2['rt_ms'],
            order_3['rt_ms']
        )

        results[freq] = {
            'x_position': {
                'f_stat': f_stat_x,
                'p_value': p_value_x,
                'significant': p_value_x < 0.05
            },
            'y_position': {
                'f_stat': f_stat_y,
                'p_value': p_value_y,
                'significant': p_value_y < 0.05
            },
            'reaction_time': {
                'f_stat': f_stat_rt,
                'p_value': p_value_rt,
                'significant': p_value_rt < 0.05
            }
        }

    return results


def calculate_position_drift(df):
    
    #Calculate how much positions change from 1st to 3rd trial
    #for each participant and frequency
    
    df = add_trial_order(df)

    drift_data = []

    for participant in df['participant_id'].unique():
        p_data = df[df['participant_id'] == participant]

        for freq in p_data['frequency'].unique():
            freq_data = p_data[p_data['frequency'] == freq]

            # Get first and last trials
            trial_1 = freq_data[freq_data['trial_order'] == 1]
            trial_3 = freq_data[freq_data['trial_order'] == 3]

            if len(trial_1) > 0 and len(trial_3) > 0:
                x1, y1 = trial_1['click_x_norm'].iloc[0], trial_1['click_y_norm'].iloc[0]
                x3, y3 = trial_3['click_x_norm'].iloc[0], trial_3['click_y_norm'].iloc[0]

                # Euclidean distance between first and last click; max possible value is √2 ≈ 1.41 (corner to corner)
                drift = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)

                drift_data.append({
                    'participant_id': participant,
                    'frequency': freq,
                    'x_drift': x3 - x1,
                    'y_drift': y3 - y1,
                    'total_drift': drift,
                    'x1': x1, 'y1': y1,
                    'x3': x3, 'y3': y3
                })

    return pd.DataFrame(drift_data)


def plot_positions_by_trial_order(df, output_path=None):
    
    #Create scatter plots showing positions at each trial order
    
    df = add_trial_order(df)
    frequencies = sorted(df['frequency'].unique())
    n_freq = len(frequencies)

    fig, axes = plt.subplots(1, n_freq, figsize=(5*n_freq, 5))

    if n_freq == 1:
        axes = [axes]

    colors = {1: 'lightblue', 2: 'orange', 3: 'darkred'}
    markers = {1: 'o', 2: 's', 3: '^'}

    for idx, freq in enumerate(frequencies):
        freq_data = df[df['frequency'] == freq]

        # Plot each trial order
        for order in [1, 2, 3]:
            order_data = freq_data[freq_data['trial_order'] == order]

            if len(order_data) > 0:
                axes[idx].scatter(
                    order_data['click_x_norm'],
                    order_data['click_y_norm'],
                    alpha=0.5, s=80,
                    color=colors[order],
                    marker=markers[order],
                    label=f'{order}. reize',
                    edgecolors='black', linewidths=0.5
                )

                # Add mean position for this order
                mean_x = order_data['click_x_norm'].mean()
                mean_y = order_data['click_y_norm'].mean()
                axes[idx].scatter(mean_x, mean_y,
                                color=colors[order], s=300,
                                marker='X', linewidths=3,
                                edgecolors='black', zorder=10)

        axes[idx].set_xlim(0, 1)
        axes[idx].set_ylim(0, 1)
        axes[idx].set_xlabel('X pozīcija (normalizēta)', fontsize=11)
        axes[idx].set_ylabel('Y pozīcija (normalizēta)', fontsize=11)
        axes[idx].set_title(f'{freq} Hz', fontsize=12)
        axes[idx].invert_yaxis()
        axes[idx].legend(loc='upper right')
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Pozīciju izmaiņas pa mēģinājumu secību\n(1., 2., 3. reize dzirdot frekvenci)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()
    return fig


def plot_mean_trajectories(df, output_path=None):
    
    #Plot how mean positions change from 1st to 3rd trial for each frequency
    
    df = add_trial_order(df)
    summary = calculate_position_by_trial_order(df)

    frequencies = sorted(df['frequency'].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    colors = plt.cm.viridis(np.linspace(0, 1, len(frequencies)))

    # X position trajectory
    for idx, freq in enumerate(frequencies):
        freq_summary = summary[summary['frequency'] == freq]
        ax1.plot(freq_summary['trial_order'], freq_summary['mean_x'],
                'o-', color=colors[idx], linewidth=3, markersize=14,
                label=f'{freq} Hz')
        ax1.errorbar(freq_summary['trial_order'], freq_summary['mean_x'],
                    yerr=freq_summary['std_x'], color=colors[idx],
                    alpha=0.3, capsize=8, elinewidth=2)

        # Add value labels
        for trial, mean_val in zip(freq_summary['trial_order'], freq_summary['mean_x']):
            ax1.annotate(f'{mean_val:.2f}',
                        xy=(trial, mean_val),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=8)

    ax1.set_xlabel('Mēģinājuma numurs', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Vidējā X pozīcija\n(0=kreisā, 1=labā)', fontsize=13, fontweight='bold')
    ax1.set_title('Horizontālā pozīcija: Vai tā mainās?\n(1. pret 2. pret 3. mēģinājumu)',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(['1. reize', '2. reize', '3. reize'], fontsize=11)
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.set_ylim(-0.05, 1.15)

    # Y position trajectory
    for idx, freq in enumerate(frequencies):
        freq_summary = summary[summary['frequency'] == freq]
        ax2.plot(freq_summary['trial_order'], freq_summary['mean_y'],
                'o-', color=colors[idx], linewidth=3, markersize=14,
                label=f'{freq} Hz')
        ax2.errorbar(freq_summary['trial_order'], freq_summary['mean_y'],
                    yerr=freq_summary['std_y'], color=colors[idx],
                    alpha=0.3, capsize=8, elinewidth=2)

        # Add value labels
        for trial, mean_val in zip(freq_summary['trial_order'], freq_summary['mean_y']):
            ax2.annotate(f'{mean_val:.2f}',
                        xy=(trial, mean_val),
                        xytext=(0, -12), textcoords='offset points',
                        ha='center', fontsize=8)

    ax2.set_xlabel('Mēģinājuma numurs', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Vidējā Y pozīcija\n(0=augša, 1=apakša)', fontsize=13, fontweight='bold')
    ax2.set_title('Vertikālā pozīcija: Vai tā mainās?\n(1. pret 2. pret 3. mēģinājumu)',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(['1. reize', '2. reize', '3. reize'], fontsize=11)
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.set_ylim(-0.05, 1.15)
    ax2.invert_yaxis()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()
    return fig


def plot_drift_analysis(df, output_path=None):
    
    #Visualize how much positions drift from 1st to 3rd trial
    
    drift_data = calculate_position_drift(df)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Drift by frequency
    sns.boxplot(data=drift_data, x='frequency', y='total_drift', ax=axes[0, 0])
    axes[0, 0].set_xlabel('Frekvence (Hz)', fontsize=12)
    axes[0, 0].set_ylabel('Pozīcijas novirze (Eiklīda attālums)', fontsize=12)
    axes[0, 0].set_title('Kopējā pozīcijas novirze: 1. līdz 3. mēģinājumam', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Distribution of drift
    axes[0, 1].hist(drift_data['total_drift'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(drift_data['total_drift'].mean(), color='red',
                      linestyle='--', linewidth=2,
                      label=f"Vidējais: {drift_data['total_drift'].mean():.3f}")
    axes[0, 1].axvline(drift_data['total_drift'].median(), color='orange',
                      linestyle='--', linewidth=2,
                      label=f"Mediāna: {drift_data['total_drift'].median():.3f}")
    axes[0, 1].set_xlabel('Pozīcijas novirze', fontsize=12)
    axes[0, 1].set_ylabel('Skaits', fontsize=12)
    axes[0, 1].set_title('Pozīcijas novirzes sadalījums', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # X drift by frequency
    sns.boxplot(data=drift_data, x='frequency', y='x_drift', ax=axes[1, 0])
    axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Frekvence (Hz)', fontsize=12)
    axes[1, 0].set_ylabel('X novirze (3. - 1.)', fontsize=12)
    axes[1, 0].set_title('Horizontālā novirze pēc frekvences', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Y drift by frequency
    sns.boxplot(data=drift_data, x='frequency', y='y_drift', ax=axes[1, 1])
    axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Frekvence (Hz)', fontsize=12)
    axes[1, 1].set_ylabel('Y novirze (3. - 1.)', fontsize=12)
    axes[1, 1].set_title('Vertikālā novirze pēc frekvences', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()
    return fig


def plot_reaction_time_by_trial_order(df, output_path=None):
    
    #Show if reaction times change across trial orders
    
    df = add_trial_order(df)

    fig, ax = plt.subplots(figsize=(10, 6))

    df_plot = df.copy()
    df_plot['trial_order'] = df_plot['trial_order'].astype(str)

    sns.boxplot(data=df_plot, x='trial_order', y='rt_ms', hue='frequency', ax=ax)

    ax.set_xlabel('Trial Order', fontsize=12)
    ax.set_ylabel('Reaction Time (ms)', fontsize=12)
    ax.set_title('Reaction Times: Does Speed Change with Repetition?', fontsize=14)
    ax.set_xticklabels(['1st time', '2nd time', '3rd time'])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(title='Frequency (Hz)')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()
    return fig


def generate_trial_order_report(df):
    
    #Generate complete report on trial order effects
    
    print("=" * 70)
    print("TRIAL ORDER ANALYSIS - Learning and Consistency Effects")
    print("=" * 70)

    # Summary statistics
    print("\n1. MEAN POSITIONS BY TRIAL ORDER")
    print("-" * 70)
    summary = calculate_position_by_trial_order(df)
    print(summary.to_string(index=False))

    # Statistical tests
    print("\n2. STATISTICAL TESTS - Do positions change across repetitions?")
    print("-" * 70)
    results = test_trial_order_effect(df)

    for freq, tests in results.items():
        print(f"\n{freq} Hz:")
        print(f"  X Position: F={tests['x_position']['f_stat']:.3f}, "
              f"p={tests['x_position']['p_value']:.4f}, "
              f"Significant: {tests['x_position']['significant']}")
        print(f"  Y Position: F={tests['y_position']['f_stat']:.3f}, "
              f"p={tests['y_position']['p_value']:.4f}, "
              f"Significant: {tests['y_position']['significant']}")
        print(f"  Reaction Time: F={tests['reaction_time']['f_stat']:.3f}, "
              f"p={tests['reaction_time']['p_value']:.4f}, "
              f"Significant: {tests['reaction_time']['significant']}")

    # Drift analysis
    print("\n3. POSITION DRIFT ANALYSIS (1st to 3rd trial)")
    print("-" * 70)
    drift_data = calculate_position_drift(df)
    drift_summary = drift_data.groupby('frequency').agg({
        'total_drift': ['mean', 'std', 'median'],
        'x_drift': ['mean', 'std'],
        'y_drift': ['mean', 'std']
    }).reset_index()
    print(drift_summary.to_string(index=False))

    print("\nOverall drift statistics:")
    print(f"  Mean drift: {drift_data['total_drift'].mean():.4f}")
    print(f"  Median drift: {drift_data['total_drift'].median():.4f}")
    print(f"  Max drift: {drift_data['total_drift'].max():.4f}")

    print("\n" + "=" * 70)

    return {
        'summary': summary,
        'statistical_tests': results,
        'drift_data': drift_data
    }


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

        print("\nGenerating trial order analysis...")
        results = generate_trial_order_report(combined_df)

        print("\nCreating visualizations...")

        plot_positions_by_trial_order(
            combined_df,
            figures_folder / "positions_by_trial_order.png"
        )

        plot_mean_trajectories(
            combined_df,
            figures_folder / "mean_trajectories.png"
        )

        plot_drift_analysis(
            combined_df,
            figures_folder / "drift_analysis.png"
        )

        plot_reaction_time_by_trial_order(
            combined_df,
            figures_folder / "rt_by_trial_order.png"
        )

        # Save results
        results['summary'].to_csv(results_folder / "trial_order_summary.csv", index=False)
        results['drift_data'].to_csv(results_folder / "position_drift.csv", index=False)

        print("\nTrial order analysis complete!")
    else:
        print(f"Data folder not found: {data_folder}")
