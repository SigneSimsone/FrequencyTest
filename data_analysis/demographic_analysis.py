# Demographic Analysis for Spatial Mapping Experiment
# Analyzes participant demographics and creates summary visualizations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from data_formatting import load_all_data


def analyze_demographics(participants_df):
    
    #Generate demographic summary statistics
    
    print("=" * 70)
    print("DEMOGRAPHIC SUMMARY")
    print("=" * 70)

    # Total participants
    print(f"\nTotal participants: {len(participants_df)}")

    # Age statistics
    if 'age' in participants_df.columns:
        print(f"\nAge statistics:")
        print(f"  Mean age: {participants_df['age'].mean():.1f}")
        print(f"  Median age: {participants_df['age'].median():.1f}")
        print(f"  Age range: {participants_df['age'].min():.0f} - {participants_df['age'].max():.0f}")
        print(f"  Std dev: {participants_df['age'].std():.1f}")

    # Gender distribution
    if 'gender' in participants_df.columns:
        print(f"\nGender distribution:")
        gender_counts = participants_df['gender'].value_counts()
        for gender, count in gender_counts.items():
            percentage = (count / len(participants_df)) * 100
            print(f"  {gender}: {count} ({percentage:.1f}%)")

    # Hearing status
    if 'hearing' in participants_df.columns:
        print(f"\nHearing status:")
        hearing_counts = participants_df['hearing'].value_counts()
        for status, count in hearing_counts.items():
            percentage = (count / len(participants_df)) * 100
            print(f"  {status}: {count} ({percentage:.1f}%)")

    print("=" * 70)


def compare_performance_by_demographics(combined_df, participants_df, output_path=None):
    """
    Compare spatial mapping performance by demographic groups
    """
    # Merge data
    df = combined_df.merge(participants_df[['participant_id', 'age', 'gender']],
                          on='participant_id', suffixes=('', '_demo'))

    # Create age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 100],
                             labels=['18-25', '26-35', '35+'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Reaction time by age group
    df_plot = df.dropna(subset=['age_group'])
    sns.boxplot(data=df_plot, x='age_group', y='rt_ms', ax=axes[0, 0])
    axes[0, 0].set_title('Reaction Time by Age Group')
    axes[0, 0].set_ylabel('Reaction Time (ms)')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Reaction time by gender
    df_plot = df.dropna(subset=['gender'])
    sns.boxplot(data=df_plot, x='gender', y='rt_ms', ax=axes[0, 1])
    axes[0, 1].set_title('Reaction Time by Gender')
    axes[0, 1].set_ylabel('Reaction Time (ms)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Spatial variance by age group
    variance_by_age = df.groupby(['age_group', 'participant_id']).agg({
        'click_x_norm': 'std',
        'click_y_norm': 'std'
    }).reset_index()
    variance_by_age['spatial_std'] = (
        variance_by_age['click_x_norm']**2 + variance_by_age['click_y_norm']**2
    ) ** 0.5

    df_plot = variance_by_age.dropna(subset=['age_group'])
    sns.boxplot(data=df_plot, x='age_group', y='spatial_std', ax=axes[1, 0])
    axes[1, 0].set_title('Spatial Consistency by Age Group')
    axes[1, 0].set_ylabel('Spatial Std Dev')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Spatial variance by gender
    variance_by_gender = df.groupby(['gender', 'participant_id']).agg({
        'click_x_norm': 'std',
        'click_y_norm': 'std'
    }).reset_index()
    variance_by_gender['spatial_std'] = (
        variance_by_gender['click_x_norm']**2 + variance_by_gender['click_y_norm']**2
    ) ** 0.5

    df_plot = variance_by_gender.dropna(subset=['gender'])
    sns.boxplot(data=df_plot, x='gender', y='spatial_std', ax=axes[1, 1])
    axes[1, 1].set_title('Spatial Consistency by Gender')
    axes[1, 1].set_ylabel('Spatial Std Dev')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

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

        print("\nAnalyzing demographics...")
        analyze_demographics(participants_df)

        print("\nCreating demographic visualizations...")

        # Performance comparison
        compare_performance_by_demographics(
            combined_df,
            participants_df,
            output_folder / "performance_by_demographics.png"
        )

        print("\nDemographic analysis complete!")
    else:
        print(f"Data folder not found: {data_folder}")
