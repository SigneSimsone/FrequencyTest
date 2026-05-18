# Main Analysis Runner
# Runs all analysis scripts and generates complete report

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from data_formatting import load_all_data
from visualizations import (
    create_scatter_plot_by_frequency,
    create_heatmap_by_frequency,
    create_combined_scatter,
    create_position_trends
)
from averaged_positions import (
    plot_averaged_scatter_by_frequency,
    plot_averaged_combined_scatter,
    plot_sharp_heatmap_by_frequency,
    plot_variance_by_participant
)
from statistical_analysis import generate_full_report
from demographic_analysis import (
    analyze_demographics,
    compare_performance_by_demographics
)
from trial_order_analysis import (
    generate_trial_order_report,
    plot_positions_by_trial_order,
    plot_mean_trajectories,
    plot_drift_analysis
)


def run_complete_analysis(data_folder, output_folder=None):
    
    #Run complete analysis pipeline

    #Parameters:
    #data_folder : str or Path
    #Path to folder containing CSV files
    #output_folder : str or Path, optional
    #Path to save outputs (default: data_analysis/output)
    
    data_folder = Path(data_folder)

    if output_folder is None:
        output_folder = Path(__file__).parent / "output"

    output_folder = Path(output_folder)
    figures_folder = output_folder / "figures"
    results_folder = output_folder / "results"

    # Create output directories
    figures_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SPATIAL MAPPING EXPERIMENT - COMPLETE ANALYSIS")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data...")
    combined_df, participants_df = load_all_data(data_folder)


    # Demographics
    print("\n[2/5] Analyzing demographics...")
    analyze_demographics(participants_df)

    compare_performance_by_demographics(
        combined_df, participants_df,
        figures_folder / "performance_by_demographics.png"
    )

    # Visualizations
    print("\n[3/5] Creating visualizations...")

    # Original visualizations (all individual clicks)
    create_scatter_plot_by_frequency(
        combined_df,
        figures_folder / "scatter_by_frequency.png"
    )

    create_heatmap_by_frequency(
        combined_df,
        figures_folder / "heatmap_by_frequency.png"
    )

    create_combined_scatter(
        combined_df,
        figures_folder / "combined_scatter.png"
    )

    create_position_trends(
        combined_df,
        figures_folder / "position_trends.png"
    )

    # Averaged positions (one point per person per frequency)
    print("  Creating averaged position visualizations...")
    plot_averaged_scatter_by_frequency(
        combined_df,
        figures_folder / "averaged_scatter_by_frequency.png"
    )

    plot_averaged_combined_scatter(
        combined_df,
        figures_folder / "averaged_combined_scatter.png"
    )

    plot_sharp_heatmap_by_frequency(
        combined_df,
        figures_folder / "sharp_heatmap_averaged.png"
    )

    plot_variance_by_participant(
        combined_df,
        figures_folder / "participant_consistency_plot.png"
    )

    print(f"  ✓ All visualizations saved to {figures_folder}")

    # Statistical analysis
    print("\n[4/5] Performing statistical analysis...")
    generate_full_report(combined_df, output_folder=results_folder)

    print(f"  ✓ Statistical results saved to {results_folder}")

    # Trial order analysis
    print("\n[5/5] Analyzing trial order effects (1st, 2nd, 3rd repetition)...")
    trial_order_results = generate_trial_order_report(combined_df)

    # Create trial order visualizations
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

    # Save trial order results
    trial_order_results['summary'].to_csv(results_folder / "trial_order_summary.csv", index=False)
    trial_order_results['drift_data'].to_csv(results_folder / "position_drift.csv", index=False)

    print(f"  ✓ Trial order analysis saved")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {output_folder}")
    print(f"  - Figures: {figures_folder}")
    print(f"  - Statistical results: {results_folder}")

    return combined_df, participants_df


if __name__ == "__main__":
    # Default paths
    data_folder = Path(__file__).parent.parent / "data"

    # Check if data folder exists
    if not data_folder.exists():
        print(f"ERROR: Data folder not found at {data_folder}")
        print("\nPlease create a 'data' folder and add your CSV files there.")
        print("Example structure:")
        print("  jsPsych/")
        print("    data/")
        print("      participant1.csv")
        print("      participant2.csv")
        print("      ...")
        sys.exit(1)

    # Run analysis
    try:
        run_complete_analysis(data_folder)
    except Exception as e:
        print(f"\nERROR: Analysis failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
