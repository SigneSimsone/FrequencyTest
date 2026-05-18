# Demographics Summary
# Calculates participant demographics: N, gender, age (M, SD, range)

import pandas as pd
import numpy as np
from pathlib import Path
from data_formatting import load_all_data


def calculate_demographics(participants_df, output_folder=None):
    
    # Calculate and print demographic summary. save to CSV.
    
    df = participants_df.copy()
    df['age'] = pd.to_numeric(df['age'], errors='coerce')  # non-numeric age entries become NaN and are excluded from stats

    n = len(df)

    # Gender
    gender_counts = df['gender'].value_counts()
    gender_rows = []
    for gender, count in gender_counts.items():
        gender_rows.append({
            'gender': gender,
            'n': count,
            'percent': round(count / n * 100, 1)
        })

    # Age
    age_stats = {
        'n': n,
        'mean': round(df['age'].mean(), 2),
        'sd': round(df['age'].std(), 2),
        'min': int(df['age'].min()),
        'max': int(df['age'].max()),
        'median': round(df['age'].median(), 2)
    }

    # Print
    print("=" * 50)
    print("DEMOGRAPHICS SUMMARY")
    print("=" * 50)
    print(f"\nTotal participants: {n}")
    print("\nGender:")
    for row in gender_rows:
        print(f"  {row['gender']}: {row['n']} ({row['percent']}%)")
    print(f"\nAge:")
    print(f"  M = {age_stats['mean']}, SD = {age_stats['sd']}")
    print(f"  Range: {age_stats['min']}–{age_stats['max']}")
    print(f"  Median: {age_stats['median']}")
    print("=" * 50)

    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        pd.DataFrame([age_stats]).to_csv(
            output_folder / "demographics_age.csv", index=False)
        pd.DataFrame(gender_rows).to_csv(
            output_folder / "demographics_gender.csv", index=False)

        print(f"\nSaved to {output_folder}")

    return age_stats, gender_rows


if __name__ == "__main__":
    data_folder = Path(__file__).parent.parent / "data"
    output_folder = Path(__file__).parent / "output" / "results"

    if data_folder.exists():
        print("Loading data...")
        _, participants_df = load_all_data(data_folder)
        calculate_demographics(participants_df, output_folder=output_folder)
    else:
        print(f"Data folder not found: {data_folder}")
