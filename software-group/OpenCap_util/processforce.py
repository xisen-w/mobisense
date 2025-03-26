import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Function to load and process IMU data
def load_imu_data(csv_file, start_time, end_time):
    df = pd.read_csv(csv_file)

    # Convert timestamp to seconds since first entry
    df["imu0_timestamp"] = pd.to_datetime(df["imu0_timestamp"])
    df["imu1_timestamp"] = pd.to_datetime(df["imu1_timestamp"])
    start_timestamp = df["imu0_timestamp"].iloc[0]

    # Zero the timestamps
    df["time"] = (df["imu0_timestamp"] - start_timestamp).dt.total_seconds()

    # Select relevant time range (18.517s to 19.017s)
    df = df[(df["time"] >= start_time) & (df["time"] <= end_time)]

    # Drop original timestamps
    df = df.drop(columns=["imu0_timestamp", "imu1_timestamp"])

    return df

# Function to load and filter ankle-related columns from .mot file
def load_ankle_mot_data(mot_file, time_range):
    with open(mot_file, 'r') as f:
        lines = f.readlines()

    # Find header index
    for i, line in enumerate(lines):
        if "endheader" in line:
            header_idx = i + 1
            break

    # Read the .mot file
    df = pd.read_csv(mot_file, delim_whitespace=True, skiprows=header_idx)

    # Identify time column
    time_col = df.columns[0]

    # Select only ankle-related columns
    ankle_cols = [col for col in df.columns if "ankle" in col.lower()]
    selected_cols = [time_col] + ankle_cols

    # Select time range
    df = df[(df[time_col] >= time_range[0]) & (df[time_col] <= time_range[1])]

    return df[selected_cols], time_col, ankle_cols

# Function to load all GRF data (excluding time column)
def load_grf_mot_data(mot_file, time_range):
    with open(mot_file, 'r') as f:
        lines = f.readlines()

    # Find header index
    for i, line in enumerate(lines):
        if "endheader" in line:
            header_idx = i + 1
            break

    # Read the .mot file
    df = pd.read_csv(mot_file, delim_whitespace=True, skiprows=header_idx)

    # Identify time column
    time_col = df.columns[0]

    # Select all columns except time
    grf_cols = [col for col in df.columns if col != time_col]

    # Select time range
    df = df[(df[time_col] >= time_range[0]) & (df[time_col] <= time_range[1])]

    return df, time_col, grf_cols

# Function to upsample IMU data **only when needed**
def upsample_imu(imu_data, target_times):
    imu_resampled = pd.DataFrame({"time": target_times})

    for col in imu_data.columns:
        if col != "time":
            if len(imu_data) >= len(target_times):
                # Downsampling: Keep nearest values
                imu_resampled[col] = np.interp(target_times, imu_data["time"], imu_data[col])
            else:
                # Upsampling: Use linear interpolation
                interp_func = interp1d(imu_data["time"], imu_data[col], kind='linear', fill_value="extrapolate")
                imu_resampled[col] = interp_func(target_times)

    return imu_resampled

# File paths
imu_csv = "2025-03-12_10-25-01-r4-walking3.csv"
force_mot = "015964a1-b2ef-4c73-9ff7-d0b6be3a836f/OpenSimData/Dynamics/walk3/kinetics_walk3_ankle_dynamics.mot"
grf_mot = "015964a1-b2ef-4c73-9ff7-d0b6be3a836f/OpenSimData/Dynamics/walk3/GRF_resultant_walk3_ankle_dynamics.mot"

# Load data
imu_data = load_imu_data(imu_csv, start_time=10-6.65, end_time=12-6.65)

# Adjust IMU time to align with ankle data's time range (0.3 to 0.8)
imu_data["time"] = imu_data["time"] - (10-6.65) + 10  # Shift IMU time to 0.3-0.8

ankle_forces, force_time_col, force_cols = load_ankle_mot_data(force_mot, time_range=(10, 12))
grf_data, grf_time_col, grf_cols = load_grf_mot_data(grf_mot, time_range=(10, 12))

# Resample IMU data **only when necessary**
imu_upsampled = upsample_imu(imu_data, ankle_forces[force_time_col])

# Combine all data into final dataset
synced_data = pd.concat([ankle_forces, grf_data.drop(columns=[grf_time_col]), imu_upsampled.drop(columns=["time"])], axis=1)

# Save to CSV
output_csv = "synced_IMU_forces_grf_fixed.csv"
synced_data.to_csv(output_csv, index=False)
print(f"Synchronized data saved to {output_csv}")

# Plot results
plt.figure(figsize=(12, 10))

# Plot ankle forces
plt.subplot(3, 1, 1)
for col in force_cols:
    plt.plot(synced_data[force_time_col], synced_data[col], label=f"Ankle Force - {col}")
plt.title("Resampled Ankle Forces")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.legend()

# Plot GRF data
plt.subplot(3, 1, 2)
for col in grf_cols:
    plt.plot(synced_data[force_time_col], synced_data[col], label=f"GRF - {col}")
plt.title("Resampled GRF Data")
plt.xlabel("Time (s)")
plt.ylabel("GRF (N)")
plt.legend()

# Plot IMU data
plt.subplot(3, 1, 3)
imu_cols = [col for col in synced_data.columns if col.startswith("imu")]
for col in imu_cols:
    plt.plot(synced_data[force_time_col], synced_data[col], label=f"IMU - {col}")
plt.title("Fixed IMU Upsampling")
plt.xlabel("Time (s)")
plt.ylabel("IMU Signals")
plt.legend()

plt.tight_layout()
plt.show()
