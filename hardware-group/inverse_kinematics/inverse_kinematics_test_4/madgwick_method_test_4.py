import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation as R
from datetime import datetime
import os

def load_imu_data(csv_file):
    """Loads IMU data from a CSV file."""
    df = pd.read_csv(csv_file)
    return df

def moving_average(data, window_size):
    """Applies a moving average filter to smooth the data."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def process_imu_data(df, beta):
    """Processes IMU data using Madgwick filter with a given beta and extracts pitch angles for both IMUs."""
    madgwick = Madgwick(beta=beta)
    
    imu0_pitch = []
    imu1_pitch = []
    timestamps = []
    dorsiflexion_angles = []

    q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # Convert timestamps to relative seconds
    initial_time = datetime.fromisoformat(df.loc[0, 'imu0_timestamp']).timestamp()

    for i in range(len(df)):
        # Read accelerometer and gyroscope data for IMU 0
        accel0 = np.array([df.loc[i, 'imu0_acc_x'], df.loc[i, 'imu0_acc_y'], df.loc[i, 'imu0_acc_z']])
        gyro0 = np.radians(np.array([df.loc[i, 'imu0_gyro_x'], df.loc[i, 'imu0_gyro_y'], df.loc[i, 'imu0_gyro_z']]))
        
        # Compute orientation using Madgwick
        q0 = madgwick.updateIMU(q0, gyr=gyro0, acc=accel0)
        r0 = R.from_quat(q0)
        _, pitch0, _ = r0.as_euler('xyz', degrees=True)
        imu0_pitch.append(pitch0)
        
        # Read accelerometer and gyroscope data for IMU 1
        accel1 = np.array([df.loc[i, 'imu1_acc_x'], df.loc[i, 'imu1_acc_y'], df.loc[i, 'imu1_acc_z']])
        gyro1 = np.radians(np.array([df.loc[i, 'imu1_gyro_x'], df.loc[i, 'imu1_gyro_y'], df.loc[i, 'imu1_gyro_z']]))
        
        # Compute orientation using Madgwick
        q1 = madgwick.updateIMU(q1, gyr=gyro1, acc=accel1)
        r1 = R.from_quat(q1)
        _, pitch1, _ = r1.as_euler('xyz', degrees=True)
        imu1_pitch.append(pitch1)
        
        # Compute dorsiflexion angle
        dorsiflexion_angle = pitch1 + pitch0 - 10
        dorsiflexion_angles.append(dorsiflexion_angle)
        
        # Convert timestamp to seconds relative to the first timestamp
        current_time = datetime.fromisoformat(df.loc[i, 'imu0_timestamp']).timestamp()
        timestamps.append(current_time - initial_time)
    
    return timestamps, imu0_pitch, imu1_pitch, dorsiflexion_angles

def plot_results(timestamps, raw_data, smoothed_data, label):
    """Plots raw and smoothed data."""
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps[:len(raw_data)], raw_data, label=f'Raw {label}', alpha=0.5)
    plt.plot(timestamps[:len(smoothed_data)], smoothed_data, label=f'Smoothed {label}', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel(f"{label} (degrees)")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    csv_file = "/Users/francescobalanzoni/Documents/Python/MEng/3YP/mobisense/software-group/data-working/assets/mar12exp/2025-03-12_10-13-30-r1-walking1.csv"
    df = load_imu_data(csv_file)
    
    beta_values = [0.001, 0.01, 0.1]
    window_sizes = [5, 10, 20, 100]
    
    for beta in beta_values:
        timestamps, imu0_pitch, imu1_pitch, dorsiflexion_angles = process_imu_data(df, beta)
        
        for window_size in window_sizes:
            smoothed_imu0 = moving_average(imu0_pitch, window_size)
            smoothed_imu1 = moving_average(imu1_pitch, window_size)
            smoothed_dorsiflexion = moving_average(dorsiflexion_angles, window_size)
            
            print(f"Plotting for beta={beta}, window_size={window_size}")
            plot_results(timestamps, imu0_pitch, smoothed_imu0, "IMU 0 Pitch")
            plot_results(timestamps, imu1_pitch, smoothed_imu1, "IMU 1 Pitch")
            plot_results(timestamps, dorsiflexion_angles, smoothed_dorsiflexion, "Dorsiflexion Angle")

if __name__ == "__main__":
    main()
