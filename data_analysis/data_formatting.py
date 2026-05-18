# Data Formatting Script for Spatial Mapping Experiment
# Loads and processes CSV files from the experiment

import pandas as pd
import json
import os
from pathlib import Path


def load_single_csv(filepath):
    #Load a single CSV file and parse demographics
    df = pd.read_csv(filepath)

    # Parse demographics JSON
    if 'demographics' in df.columns and not df['demographics'].isna().all():
        demo_json = df['demographics'].iloc[0]
        demographics = json.loads(demo_json)

        # Add demographic columns
        df['gender'] = demographics.get('gender', None)
        df['age'] = demographics.get('age', None)
        df['hearing'] = demographics.get('hearing', None)
        df['hobbies'] = str(demographics.get('hobbies', []))  # list → string so it serializes to a single CSV cell

    return df


def load_all_data(data_folder):

    #Load all CSV files from a folder and combine them

    #Parameters:
    #data_folder : str or Path
    #Path to folder containing CSV files

    #Returns:
    #combined_df : DataFrame
    #Combined data from all participants
    #participants_df : DataFrame
    #Summary of participant demographics
  
    data_folder = Path(data_folder)
    csv_files = list(data_folder.glob('*.csv'))

    if not csv_files:
        raise ValueError(f"No CSV files found in {data_folder}")

    print(f"Found {len(csv_files)} CSV files")

    all_data = []
    participant_info = []

    for csv_file in csv_files:
        try:
            df = load_single_csv(csv_file)
            all_data.append(df)

            # Extract participant info
            if len(df) > 0:
                participant_info.append({
                    'participant_id': df['participant_id'].iloc[0],
                    'gender': df['gender'].iloc[0] if 'gender' in df.columns else None,
                    'age': df['age'].iloc[0] if 'age' in df.columns else None,
                    'hearing': df['hearing'].iloc[0] if 'hearing' in df.columns else None,
                    'n_trials': len(df)
                })

            print(f"Loaded: {csv_file.name}")
        except Exception as e:
            print(f"Error loading {csv_file.name}: {e}")

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)  # ignore_index prevents duplicate row numbers across participants
    participants_df = pd.DataFrame(participant_info)

    print(f"\nTotal participants: {len(participants_df)}")
    print(f"Total trials: {len(combined_df)}")

    return combined_df, participants_df


def get_frequency_data(df, frequency):
    #Extract data for a specific frequency
    return df[df['frequency'] == frequency].copy()


def calculate_mean_positions(df):

    #Calculate mean position for each frequency
    #Returns DataFrame with mean x, y positions and std dev

    summary = df.groupby('frequency').agg({
        'click_x_norm': ['mean', 'std', 'count'],
        'click_y_norm': ['mean', 'std'],
        'rt_ms': ['mean', 'median']
    }).reset_index()

    # Flatten column names
    summary.columns = ['frequency', 'mean_x', 'std_x', 'count',
                       'mean_y', 'std_y', 'mean_rt', 'median_rt']

    return summary


if __name__ == "__main__":
    # Example usage
    data_folder = Path(__file__).parent.parent / "data"  # Adjust path as needed

    if data_folder.exists():
        combined_df, participants_df = load_all_data(data_folder)

        # Save processed data
        combined_df.to_csv(data_folder / "combined_data.csv", index=False)
        participants_df.to_csv(data_folder / "participants_summary.csv", index=False)

        # Calculate summary statistics
        summary = calculate_mean_positions(combined_df)
        print("\nMean positions by frequency:")
        print(summary)
        summary.to_csv(data_folder / "position_summary.csv", index=False)
    else:
        print(f"Data folder not found: {data_folder}")
        print("Please create a 'data' folder and add your CSV files there.")
