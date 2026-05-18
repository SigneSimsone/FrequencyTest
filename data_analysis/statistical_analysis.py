# Statistical Analysis for Spatial Mapping Experiment
# Non-parametric methods for repeated-measures design:
# - Friedman test
# - Kendall's W
# - Wilcoxon signed-rank tests (pairwise)
# - Spearman correlation
# - Cohen's d (paired)

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from data_formatting import load_all_data


def calculate_averaged_positions(df):
    
    #Average the 3 repetitions per participant per frequency.
    #Returns one row per participant per frequency.
    
    averaged = df.groupby(['participant_id', 'frequency']).agg({
        'click_x_norm': 'mean',
        'click_y_norm': 'mean'
    }).reset_index()
    averaged.columns = ['participant_id', 'frequency', 'avg_x', 'avg_y']
    return averaged


def descriptive_statistics(averaged_df):
    
    #Mean and SD of averaged positions per frequency.
    
    summary = averaged_df.groupby('frequency').agg(
        mean_y=('avg_y', 'mean'),
        std_y=('avg_y', 'std'),
        mean_x=('avg_x', 'mean'),
        std_x=('avg_x', 'std'),
        n=('avg_y', 'count')
    ).reset_index()
    return summary


def pivot_by_frequency(averaged_df, axis):
    
    #Pivot so each column is one frequency, each row is one participant.
    #axis: 'avg_y' or 'avg_x'
    
    pivoted = averaged_df.pivot(index='participant_id', columns='frequency', values=axis)
    return pivoted


def friedman_test(averaged_df, axis):
    
    #Friedman test across the three frequency conditions.
    #axis: 'avg_y' or 'avg_x'
    #Non-parametric alternative to repeated-measures ANOVA; appropriate for
    #small N and non-normally distributed click positions.
    
    pivoted = pivot_by_frequency(averaged_df, axis)
    frequencies = sorted(pivoted.columns)
    groups = [pivoted[f].values for f in frequencies]
    chi2, p = stats.friedmanchisquare(*groups)
    return chi2, p, frequencies


def kendall_w(averaged_df, axis):
    
    #Kendall's W (concordance coefficient) derived from the Friedman statistic.
    #W = chi2 / (N * (k - 1))
    #where N = number of participants, k = number of conditions.
    
    pivoted = pivot_by_frequency(averaged_df, axis)
    n = len(pivoted)        # participants
    k = len(pivoted.columns)  # conditions
    groups = [pivoted[f].values for f in sorted(pivoted.columns)]
    chi2, _ = stats.friedmanchisquare(*groups)
    w = chi2 / (n * (k - 1))
    return w


def wilcoxon_pairwise(averaged_df, axis):
    
    #Wilcoxon signed-rank tests for all three frequency pairs.
    #axis: 'avg_y' or 'avg_x'
    #Returns list of dicts with freq1, freq2, statistic, p_value, cohens_d.
    
    pivoted = pivot_by_frequency(averaged_df, axis)
    frequencies = sorted(pivoted.columns)
    results = []

    pairs = [
        (frequencies[0], frequencies[1]),
        (frequencies[1], frequencies[2]),
        (frequencies[0], frequencies[2]),
    ]

    for f1, f2 in pairs:
        a = pivoted[f1].values
        b = pivoted[f2].values
        stat, p = stats.wilcoxon(a, b)

        # Paired Cohen's d: mean(diff) / std(diff); ddof=1 uses sample (not population) std
        diff = a - b
        d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0

        results.append({
            'freq1': f1,
            'freq2': f2,
            'statistic': stat,
            'p_value': p,
            'cohens_d': abs(d)
        })

    return results


def spearman_correlation(averaged_df, axis):
    
    #Spearman correlation between frequency and position,
    #computed on participant-averaged data (one value per participant per frequency).
    #axis: 'avg_y' or 'avg_x'
    
    rho, p = stats.spearmanr(averaged_df['frequency'], averaged_df[axis])
    return rho, p


def generate_full_report(df, output_folder=None):
    
    #Run all analyses, print a comprehensive report, and optionally save to CSV.
    
    averaged_df = calculate_averaged_positions(df)

    print("=" * 70)
    print("STATISTICAL ANALYSIS REPORT")
    print("=" * 70)

    # 1. Descriptive statistics
    print("\n1. DESCRIPTIVE STATISTICS (participant averages)")
    print("-" * 70)
    desc = descriptive_statistics(averaged_df)
    for _, row in desc.iterrows():
        print(f"  {int(row['frequency'])} Hz  "
              f"Y: M={row['mean_y']:.3f} SD={row['std_y']:.3f}  "
              f"X: M={row['mean_x']:.3f} SD={row['std_x']:.3f}  "
              f"N={int(row['n'])}")

    omnibus_rows = []
    pairwise_rows = []

    for axis_label, axis_col in [("Y (vertical)", "avg_y"), ("X (horizontal)", "avg_x")]:
        print(f"\n{'=' * 70}")
        print(f"  {axis_label.upper()} AXIS")
        print("=" * 70)

        chi2, p, freqs = friedman_test(averaged_df, axis_col)
        df_friedman = len(freqs) - 1
        print(f"\n  Friedman test: χ²({df_friedman}) = {chi2:.2f}, p = {p:.4f}")

        w = kendall_w(averaged_df, axis_col)
        print(f"  Kendall's W = {w:.3f}")

        rho, p_rho = spearman_correlation(averaged_df, axis_col)
        print(f"  Spearman ρ = {rho:.3f}, p = {p_rho:.4f}")

        omnibus_rows.append({
            'axis': axis_label,
            'friedman_chi2': round(chi2, 3),
            'friedman_df': df_friedman,
            'friedman_p': round(p, 4),
            'kendall_W': round(w, 3),
            'spearman_rho': round(rho, 3),
            'spearman_p': round(p_rho, 4)
        })

        print(f"\n  Wilcoxon pairwise tests:")
        pairwise = wilcoxon_pairwise(averaged_df, axis_col)
        for r in pairwise:
            sig = "p < .001" if r['p_value'] < 0.001 else f"p = {r['p_value']:.3f}"
            print(f"    {int(r['freq1'])} Hz vs {int(r['freq2'])} Hz: "
                  f"W = {r['statistic']:.1f}, {sig}, Cohen's d = {r['cohens_d']:.2f}")
            pairwise_rows.append({
                'axis': axis_label,
                'freq1_hz': int(r['freq1']),
                'freq2_hz': int(r['freq2']),
                'wilcoxon_W': round(r['statistic'], 1),
                'p_value': round(r['p_value'], 4),
                'cohens_d': round(r['cohens_d'], 3)
            })

    print("\n" + "=" * 70)
    print("END OF REPORT")
    print("=" * 70)

    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        desc.to_csv(output_folder / "descriptive_statistics.csv", index=False)
        pd.DataFrame(omnibus_rows).to_csv(
            output_folder / "omnibus_tests.csv", index=False)
        pd.DataFrame(pairwise_rows).to_csv(
            output_folder / "pairwise_wilcoxon.csv", index=False)
        averaged_df.to_csv(output_folder / "averaged_positions.csv", index=False)

        print(f"\nResults saved to {output_folder}")

    return averaged_df


if __name__ == "__main__":
    data_folder = Path(__file__).parent.parent / "data"
    output_folder = Path(__file__).parent / "output" / "results"
    output_folder.mkdir(parents=True, exist_ok=True)

    if data_folder.exists():
        print("Loading data...")
        combined_df, _ = load_all_data(data_folder)

        print("\nRunning statistical analysis...")
        averaged_df = generate_full_report(combined_df)

        averaged_df.to_csv(output_folder / "averaged_positions.csv", index=False)
        print(f"\nAveraged positions saved to {output_folder / 'averaged_positions.csv'}")
    else:
        print(f"Data folder not found: {data_folder}")
