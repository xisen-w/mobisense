import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from datetime import datetime
import os
from filterpy.kalman import KalmanFilter


def load_imu_data(csv_file):
    """Loads IMU data from a CSV file."""
    df = pd.read_csv(csv_file)
    return df


def initialize_kalman():
    """Initializes a Kalman filter for pitch estimation."""
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1, 1], [0, 1]])  # State transition matrix
    kf.H = np.array([[1, 0]])  # Measurement function
    kf.P *= 1000  # Covariance matrix
    kf.R = 10  # Measurement noise
    kf.Q = np.array([[1, 0], [0, 1]])  # Process noise
    return kf


def process_imu_data(df):
    """Processes IMU data using a Kalman filter and extracts pitch angles."""
    kf0 = initialize_kalman()
    kf1 = initialize_kalman()
    
    imu0_pitch = []
    imu1_pitch = []
    timestamps = []
    dorsiflexion_angles = []
    
    initial_time = datetime.fromisoformat(df.loc[0, 'imu0_timestamp']).timestamp()
    
    for i in range(len(df)):
        # Read accelerometer and gyroscope data for IMU 0
        accel0 = np.array([df.loc[i, 'imu0_acc_x'], df.loc[i, 'imu0_acc_y'], df.loc[i, 'imu0_acc_z']])
        gyro0 = np.radians(np.array([df.loc[i, 'imu0_gyro_x'], df.loc[i, 'imu0_gyro_y'], df.loc[i, 'imu0_gyro_z']]))
        
        # Compute pitch from accelerometer
        pitch0_acc = np.degrees(np.arctan2(accel0[1], accel0[2]))
        
        # Kalman filter update for IMU 0
        kf0.predict()
        kf0.update([pitch0_acc])
        imu0_pitch.append(kf0.x[0])
        
        # Read accelerometer and gyroscope data for IMU 1
        accel1 = np.array([df.loc[i, 'imu1_acc_x'], df.loc[i, 'imu1_acc_y'], df.loc[i, 'imu1_acc_z']])
        gyro1 = np.radians(np.array([df.loc[i, 'imu1_gyro_x'], df.loc[i, 'imu1_gyro_y'], df.loc[i, 'imu1_gyro_z']]))
        
        # Compute pitch from accelerometer
        pitch1_acc = np.degrees(np.arctan2(accel1[1], accel1[2]))
        
        # Kalman filter update for IMU 1
        kf1.predict()
        kf1.update([pitch1_acc])
        imu1_pitch.append(kf1.x[0])
        
        # Compute dorsiflexion angle
        dorsiflexion_angle = kf0.x[0] + kf1.x[0]
        dorsiflexion_angles.append(dorsiflexion_angle)
        
        # Convert timestamp to seconds relative to the first timestamp
        current_time = datetime.fromisoformat(df.loc[i, 'imu0_timestamp']).timestamp()
        timestamps.append(current_time - initial_time)
    
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


def main():
    csv_file = "software-group/data-working/assets/mar12exp/2025-03-12_10-13-30-r1-walking1.csv"
    df = load_imu_data(csv_file)
    
    timestamps, imu0_pitch, imu1_pitch, dorsiflexion_angles = process_imu_data(df)
    plot_pitch(timestamps, imu0_pitch, imu1_pitch)
    plot_dorsiflexion_angle(timestamps, dorsiflexion_angles)


if __name__ == "__main__":
    main()
