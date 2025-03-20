import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation as R
from datetime import datetime

def load_imu_data(csv_file):
    """Loads IMU data from a CSV file."""
    df = pd.read_csv(csv_file)
    return df

def process_imu_data(df):
    """Processes IMU data using Madgwick filter."""
    madgwick = Madgwick(beta=0.0001)
    
    imu0_pitch = []
    imu1_pitch = []
    timestamps = []

    q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    accel_data_0 = df[['imu0_acc_x', 'imu0_acc_y', 'imu0_acc_z']].values
    gyro_data_0 = np.radians(df[['imu0_gyro_x', 'imu0_gyro_y', 'imu0_gyro_z']].values)
    
    accel_data_1 = df[['imu1_acc_x', 'imu1_acc_y', 'imu1_acc_z']].values
    gyro_data_1 = np.radians(df[['imu1_gyro_x', 'imu1_gyro_y', 'imu1_gyro_z']].values)
    
    start_time = datetime.fromisoformat(df.loc[0, 'imu0_timestamp'])

    for i in range(len(df)):
        accel0 = accel_data_0[i]
        gyro0 = gyro_data_0[i]
        accel1 = accel_data_1[i]
        gyro1 = gyro_data_1[i]
        
        # Compute orientation using Madgwick
        q0 = madgwick.updateIMU(q0, gyr=gyro0, acc=accel0)
        r0 = R.from_quat(q0)
        _, pitch0, _ = r0.as_euler('xyz', degrees=True)
        
        q1 = madgwick.updateIMU(q1, gyr=gyro1, acc=accel1)
        r1 = R.from_quat(q1)
        _, pitch1, _ = r1.as_euler('xyz', degrees=True)
        
        # Convert timestamp to datetime format
        current_time = datetime.fromisoformat(df.loc[i, 'imu0_timestamp'])
        
        imu0_pitch.append(pitch0)
        imu1_pitch.append(pitch1)
        timestamps.append(current_time)
    
    return timestamps, imu0_pitch, imu1_pitch

def plot_pitch(timestamps, imu0_pitch, imu1_pitch):
    """Plots pitch angles for both IMUs over time."""
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, imu0_pitch, label='IMU 0 Pitch', color='b')
    plt.plot(timestamps, imu1_pitch, label='IMU 1 Pitch', color='r')
    plt.xlabel("Time")
    plt.ylabel("Pitch Angle (degrees)")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.show()

def plot_pitch_difference(timestamps, imu0_pitch, imu1_pitch):
    start_time = timestamps[0]
    slope = 5 / 52  # 6 degrees over 30 seconds = 0.2 degrees per second
    pitch_diff = []
    
    for i, (p0, p1) in enumerate(zip(imu0_pitch, imu1_pitch)):
        elapsed_time = (timestamps[i] - start_time).total_seconds()
        correction = slope * elapsed_time
        pitch_diff.append(p1 + p0 - correction )
    
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, pitch_diff)
    plt.xlabel("Time")
    plt.ylabel("Plantar Flexion Angle (degrees)")
    plt.grid()
    plt.xticks(rotation=45)
    plt.show()

def main():
    csv_file = "/Users/francescobalanzoni/Documents/Python/MEng/3YP/mobisense/software-group/data-working/assets/mar12exp/2025-03-12_10-22-44-r3-limping2.csv"
    df = load_imu_data(csv_file)
    
    timestamps, imu0_pitch, imu1_pitch = process_imu_data(df)
    plot_pitch(timestamps, imu0_pitch, imu1_pitch)
    plot_pitch_difference(timestamps, imu0_pitch, imu1_pitch)

if __name__ == "__main__":
    main()
