import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def extract_ankle_data(mot_file, output_csv):
    """Extracts ankle-related kinematics data from a .mot file and saves it as a CSV."""
    with open(mot_file, 'r') as f:
        lines = f.readlines()

    # Find the line index where column headers start
    for i, line in enumerate(lines):
        if "endheader" in line:
            header_idx = i + 1
            break

    # Read the file as a DataFrame starting from the column headers
    df = pd.read_csv(mot_file, delim_whitespace=True, skiprows=header_idx)

    # Identify columns relevant to the ankle
    selected_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ["ankle", "talus", "calcaneus"])]

    # Ensure time is included
    if 'time' in df.columns:
        selected_columns.insert(0, 'time')

    # Extract relevant data
    ankle_data = df[selected_columns]

    # Save to CSV
    ankle_data.to_csv(output_csv, index=False)
    print(f"Extracted ankle data saved to {output_csv}")

    return ankle_data

def apply_savgol_filter(df, window=21, polyorder=3):
    """Applies a Savitzky-Golay filter to smooth the ankle angle data."""
    smoothed_df = df.copy()
    for col in df.columns:
        if col != 'time':  # Skip time column
            smoothed_df[col] = savgol_filter(df[col], window_length=window, polyorder=polyorder)
    return smoothed_df

def plot_ankle_kinematics(raw_df, smoothed_df, save_path):
    """Plots raw and smoothed ankle kinematics data."""
    plt.figure(figsize=(12, 6))

    # Plot raw data
    plt.subplot(2, 1, 1)
    for col in raw_df.columns:
        if col != 'time':
            plt.plot(raw_df['time'], raw_df[col], label=f"Raw {col}")
    plt.title("Raw Ankle Kinematics")
    plt.xlabel("Time (s)")
    plt.ylabel("Ankle Angles (deg)")
    plt.legend()

    # Plot smoothed data
    plt.subplot(2, 1, 2)
    for col in smoothed_df.columns:
        if col != 'time':
            plt.plot(smoothed_df['time'], smoothed_df[col], label=f"Smoothed {col}")
    plt.title("Smoothed Ankle Kinematics (Savitzky-Golay)")
    plt.xlabel("Time (s)")
    plt.ylabel("Ankle Angles (deg)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")

# Usage
mot_file = "OpenCapData_53e6ae0d-ca4b-44c9-868e-462b8281a6d9/OpenSimData/Kinematics/walking3.mot" 
output_csv = "ProcessedData/walking3_ankle.csv"
save_path = "Plots/ankle_kinematics_smooth.png"

print("Extracting ankle data...")
ankle_data = extract_ankle_data(mot_file, output_csv)
print("Data extraction complete.")

print("Applying smoothing...")
smoothed_ankle_data = apply_savgol_filter(ankle_data)
print("Smoothing complete.")

print("Generating plot...")
plot_ankle_kinematics(ankle_data, smoothed_ankle_data, save_path)
print("Plot saved successfully.")