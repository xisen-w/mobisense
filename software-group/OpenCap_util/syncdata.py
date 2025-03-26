import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Mapping for usable pairs (excluding walk2)
pairs = {
    "limping3": "2025-03-12_10-27-09-r5-limping3.csv",
    "walk1": "2025-03-12_10-13-30-r1-walking1.csv",
    "walk3": "2025-03-12_10-25-01-r4-walking3.csv"
}

# Folder locations
imu_folder = "."
opencap_folder = "OpenCapData_015964a1-b2ef-4c73-9ff7-d0b6be3a836f/OpenSimData/Kinematics"
output_folder = "SyncedData"
os.makedirs(output_folder, exist_ok=True)

# OpenCap time when IMU started
imu_start_times = {
    "limping3": 7.27,
    "walk1": 3.48,
    "walk3": 6.65
}

def load_and_zero_imu(file_path):
    df = pd.read_csv(file_path)
    df["imu0_timestamp"] = pd.to_datetime(df["imu0_timestamp"])
    start = df["imu0_timestamp"].iloc[0]
    df["time"] = (df["imu0_timestamp"] - start).dt.total_seconds()
    return df.drop(columns=["imu0_timestamp", "imu1_timestamp"])

def load_opencap_ankles_only(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "endheader" in line:
            header_idx = i + 1
            break
    df = pd.read_csv(file_path, delim_whitespace=True, skiprows=header_idx)
    ankle_cols = [col for col in df.columns if "ankle" in col.lower()]
    return df[["time"] + ankle_cols]

def upsample_imu_data(imu_data, target_times):
    imu_resampled = pd.DataFrame({"time": target_times})
    for col in imu_data.columns:
        if col != "time":
            if len(imu_data) >= len(target_times):
                # Downsampling: interpolate directly
                imu_resampled[col] = np.interp(target_times, imu_data["time"], imu_data[col])
            else:
                # Upsampling: use interp1d with extrapolation
                interp_func = interp1d(imu_data["time"], imu_data[col], kind='linear', fill_value="extrapolate")
                imu_resampled[col] = interp_func(target_times)
    return imu_resampled

def plot_synced_data(opencap_df, imu_upsampled, pair_name):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot OpenCap angles
    axes[0].set_title(f"OpenCap Ankle Angles – {pair_name}")
    if "ankle_angle_r" in opencap_df.columns:
        axes[0].plot(opencap_df["time"], opencap_df["ankle_angle_r"], label="ankle_angle_r", linewidth=2)
    if "ankle_angle_l" in opencap_df.columns:
        axes[0].plot(opencap_df["time"], opencap_df["ankle_angle_l"], label="ankle_angle_l", linewidth=2)
    axes[0].legend()
    axes[0].set_ylabel("Angle (deg)")
    axes[0].grid(True)

    # Plot IMU data
    axes[1].set_title("Upsampled IMU Orientation (Roll, Pitch, Yaw)")
    for col in imu_upsampled.columns:
        if col != "time" and any(kw in col.lower() for kw in ["roll", "pitch", "yaw"]):
            axes[1].plot(imu_upsampled["time"], imu_upsampled[col], label=col, alpha=0.7)

    axes[1].legend()
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Orientation (rad or deg)")
    axes[1].grid(True)

    plt.tight_layout()
    out_path = os.path.join(output_folder, f"{pair_name}_sync_plot.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")

def synchronise_and_save(pair_name, imu_file, opencap_file, start_time):
    imu_df = load_and_zero_imu(os.path.join(imu_folder, imu_file))
    opencap_df = load_opencap_ankles_only(os.path.join(opencap_folder, opencap_file))

    # Keep IMU time as-is (relative to its start)
    # Shift OpenCap so that OpenCap = 0 at IMU start
    opencap_df["time"] = opencap_df["time"] - start_time


    # Only keep shared time range
    end_time = min(imu_df["time"].max(), opencap_df["time"].max())
    imu_df = imu_df[imu_df["time"] <= end_time]
    opencap_df = opencap_df[opencap_df["time"] >= 0]
    opencap_df = opencap_df[opencap_df["time"] <= end_time]


    imu_upsampled = upsample_imu_data(imu_df, opencap_df["time"].values)

    combined = pd.concat([opencap_df.reset_index(drop=True),
                          imu_upsampled.drop(columns=["time"]).reset_index(drop=True)], axis=1)
    out_csv = os.path.join(output_folder, f"{pair_name}_synced.csv")
    combined.to_csv(out_csv, index=False)
    print(f"Saved synced file: {out_csv}")

    plot_synced_data(opencap_df, imu_upsampled, pair_name)

# Run
for pair_name, imu_file in pairs.items():
    mot_file = f"{pair_name}.mot"
    if not os.path.exists(os.path.join(opencap_folder, mot_file)):
        print(f"Skipping {pair_name}: .mot file not found.")
        continue
    synchronise_and_save(pair_name, imu_file, mot_file, imu_start_times[pair_name])
