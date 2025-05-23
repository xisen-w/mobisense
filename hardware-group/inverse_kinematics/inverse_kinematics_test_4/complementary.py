import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ahrs.filters import Complementary
from scipy.spatial.transform import Rotation as R
from datetime import datetime
import os

def load_imu_data(csv_file):
    """Loads IMU data from a CSV file."""
    df = pd.read_csv(csv_file)
    return df

def remove_drift(signal, window_size=100):
    """Removes drift by subtracting a moving average."""
    return signal - pd.Series(signal).rolling(window=window_size, center=True, min_periods=1).mean()

def process_imu_data(df):
    """Processes IMU data using Complementary filter and extracts pitch angles for both IMUs."""
    complementary = Complementary()
    
    imu0_pitch = []
    imu1_pitch = []
    timestamps = []
    dorsiflexion_angles = []
    
    q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # Convert timestamps to relative seconds
    initial_time = datetime.fromisoformat(df.loc[0, 'imu0_timestamp']).timestamp()

    for i in range(len(df)):
        # Read accelerometer and gyroscope data for IMU 0 (convert acceleration from milli m/s^2 to m/s^2)
        accel0 = np.array([df.loc[i, 'imu0_acc_x'], df.loc[i, 'imu0_acc_y'], df.loc[i, 'imu0_acc_z']]) / 1000
        gyro0 = np.radians(np.array([df.loc[i, 'imu0_gyro_x'], df.loc[i, 'imu0_gyro_y'], df.loc[i, 'imu0_gyro_z']]))
        
        # Compute orientation using Complementary filter
        q0 = complementary.update(q0, gyr=gyro0, acc=accel0)
        r0 = R.from_quat(q0)
        _, pitch0, _ = r0.as_euler('xyz', degrees=True)
        imu0_pitch.append(pitch0)
        
        # Read accelerometer and gyroscope data for IMU 1 (convert acceleration from milli m/s^2 to m/s^2)
        accel1 = np.array([df.loc[i, 'imu1_acc_x'], df.loc[i, 'imu1_acc_y'], df.loc[i, 'imu1_acc_z']]) / 1000
        gyro1 = np.radians(np.array([df.loc[i, 'imu1_gyro_x'], df.loc[i, 'imu1_gyro_y'], df.loc[i, 'imu1_gyro_z']]))
        
        # Compute orientation using Complementary filter
        q1 = complementary.update(q1, gyr=gyro1, acc=accel1)
        r1 = R.from_quat(q1)
        _, pitch1, _ = r1.as_euler('xyz', degrees=True)
        imu1_pitch.append(pitch1)
        
        # Compute dorsiflexion angle
        dorsiflexion_angle = pitch1 + pitch0 
        dorsiflexion_angles.append(dorsiflexion_angle)
        
        # Convert timestamp to seconds relative to the first timestamp
        current_time = datetime.fromisoformat(df.loc[i, 'imu0_timestamp']).timestamp()
        timestamps.append(current_time - initial_time)

    # Remove drift using moving average
    imu0_pitch = remove_drift(imu0_pitch)
    imu1_pitch = remove_drift(imu1_pitch)
    dorsiflexion_angles = remove_drift(dorsiflexion_angles)
    
    return timestamps, imu0_pitch, imu1_pitch, dorsiflexion_angles

def plot_pitch(timestamps, imu0_pitch, imu1_pitch):
    """Plots pitch angles for both IMUs over time."""
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, imu0_pitch, label='IMU 0 Pitch', color='b')
    plt.plot(timestamps, imu1_pitch, label='IMU 1 Pitch', color='r')
    plt.xlabel("Time")
    plt.ylabel("Pitch Angle (degrees)")
    plt.legend()
    plt.grid()
    plt.show()

def plot_dorsiflexion_angle(timestamps, dorsiflexion_angles):
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, dorsiflexion_angles)
    plt.xlabel("Time")
    plt.ylabel("Dorsiflexion Angle (degrees)")
    plt.grid()
    plt.show()

def save_updated_csv(df, dorsiflexion_angles, original_csv):
    """Saves the updated CSV file with dorsiflexion angles in the 'mar12exp_updated' folder."""
    df['dorsiflexion_angle'] = dorsiflexion_angles
    
    # Define new file path
    updated_folder = os.path.join(os.path.dirname(original_csv), "mar12exp_updated")
    updated_file = os.path.join(updated_folder, os.path.basename(original_csv))
    
    # Save CSV
    df.to_csv(updated_file, index=False)
    print(f"Updated CSV saved to: {updated_file}")

def main():
    csv_file = "software-group/data-working/assets/mar12exp/2025-03-12_10-13-30-r1-walking1.csv"
    df = load_imu_data(csv_file)
    
    timestamps, imu0_pitch, imu1_pitch, dorsiflexion_angles = process_imu_data(df)
    plot_pitch(timestamps, imu0_pitch, imu1_pitch)
    plot_dorsiflexion_angle(timestamps, dorsiflexion_angles)
    # save_updated_csv(df, dorsiflexion_angles, csv_file)

if __name__ == "__main__":
    main()
