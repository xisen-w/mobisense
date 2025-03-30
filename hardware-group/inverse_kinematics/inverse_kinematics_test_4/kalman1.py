import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from datetime import datetime
import os

class KalmanFilter:
    """Simple Kalman Filter for estimating pitch from gyroscope and accelerometer data."""
    def __init__(self, dt, process_var=1e-3, measurement_var=1e-2):
        self.dt = dt  # Time step
        self.x = np.array([0.0, 0.0])  # State vector: [angle, angular velocity]
        self.P = np.eye(2)  # Covariance matrix
        self.Q = np.array([[process_var, 0], [0, process_var]])  # Process noise covariance
        self.R = measurement_var  # Measurement noise covariance
        self.H = np.array([1, 0])  # Measurement matrix
        self.F = np.array([[1, dt], [0, 1]])  # State transition matrix

    def predict(self, gyro_rate):
        """Prediction step using the gyroscope measurement."""
        self.x = self.F @ self.x + np.array([self.dt * gyro_rate, 0])
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, accel_angle):
        """Correction step using accelerometer-based angle."""
        y = accel_angle - self.H @ self.x  # Measurement residual
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T / S  # Kalman gain
        self.x = self.x + K * y
        self.P = (np.eye(2) - K @ self.H) @ self.P

def load_imu_data(csv_file):
    """Loads IMU data from a CSV file."""
    return pd.read_csv(csv_file)

def remove_drift(signal, window_size=100):
    """Removes drift by subtracting a moving average."""
    return signal - pd.Series(signal).rolling(window=window_size, center=True, min_periods=1).mean()

def process_imu_data(df):
    """Processes IMU data using a Kalman filter and extracts pitch angles for both IMUs."""
    dt = 0.3  # Assuming 30 Hz IMU data (adjust if different)
    
    kf_imu0 = KalmanFilter(dt)
    kf_imu1 = KalmanFilter(dt)

    imu0_pitch, imu1_pitch = [], []
    timestamps, dorsiflexion_angles = [], []

    initial_time = datetime.fromisoformat(df.loc[0, 'imu0_timestamp']).timestamp()

    for i in range(len(df)):
        # Read gyroscope data (convert to rad/s)
        gyro0 = np.radians(df.loc[i, 'imu0_gyro_y'])  # Y-axis for pitch
        gyro1 = np.radians(df.loc[i, 'imu1_gyro_y'])

        # Compute accelerometer-based pitch estimate
        accel0 = np.array([df.loc[i, 'imu0_acc_x'], df.loc[i, 'imu0_acc_y'], df.loc[i, 'imu0_acc_z']])
        accel_pitch0 = np.degrees(np.arctan2(accel0[0], accel0[2]))  # atan2(x, z)

        accel1 = np.array([df.loc[i, 'imu1_acc_x'], df.loc[i, 'imu1_acc_y'], df.loc[i, 'imu1_acc_z']])
        accel_pitch1 = np.degrees(np.arctan2(accel1[0], accel1[2]))

        # Apply Kalman filter
        kf_imu0.predict(gyro0)
        kf_imu0.update(accel_pitch0)
        imu0_pitch.append(kf_imu0.x[0])

        kf_imu1.predict(gyro1)
        kf_imu1.update(accel_pitch1)
        imu1_pitch.append(kf_imu1.x[0])

        # Compute dorsiflexion angle
        dorsiflexion_angle = imu1_pitch[-1] - imu0_pitch[-1]
        dorsiflexion_angles.append(dorsiflexion_angle)

        # Convert timestamp to seconds relative to the first timestamp
        current_time = datetime.fromisoformat(df.loc[i, 'imu0_timestamp']).timestamp()
        timestamps.append(current_time - initial_time)

    # Remove drift
    imu0_pitch = remove_drift(imu0_pitch)
    imu1_pitch = remove_drift(imu1_pitch)
    dorsiflexion_angles = remove_drift(dorsiflexion_angles)

    return timestamps, imu0_pitch, imu1_pitch, dorsiflexion_angles

def plot_pitch(timestamps, imu0_pitch, imu1_pitch):
    """Plots pitch angles for both IMUs over time."""
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, imu0_pitch, label='IMU 0 Pitch (Kalman)', color='b')
    plt.plot(timestamps, imu1_pitch, label='IMU 1 Pitch (Kalman)', color='r')
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch Angle (degrees)")
    plt.legend()
    plt.grid()
    plt.show()

def plot_dorsiflexion_angle(timestamps, dorsiflexion_angles):
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, dorsiflexion_angles, label="Dorsiflexion Angle", color='g')
    plt.xlabel("Time (s)")
    plt.ylabel("Dorsiflexion Angle (degrees)")
    plt.legend()
    plt.grid()
    plt.show()

def save_updated_csv(df, dorsiflexion_angles, original_csv):
    """Saves the updated CSV file with dorsiflexion angles."""
    df['dorsiflexion_angle'] = dorsiflexion_angles
    updated_folder = os.path.join(os.path.dirname(original_csv), "kalman_filtered")
    os.makedirs(updated_folder, exist_ok=True)
    updated_file = os.path.join(updated_folder, os.path.basename(original_csv))
    df.to_csv(updated_file, index=False)
    print(f"Updated CSV saved to: {updated_file}")

def main():
    csv_file = "/Users/francescobalanzoni/Documents/Python/MEng/3YP/mobisense/software-group/data-working/assets/mar12exp/2025-03-12_10-22-44-r3-limping2.csv"
    df = load_imu_data(csv_file)
    
    timestamps, imu0_pitch, imu1_pitch, dorsiflexion_angles = process_imu_data(df)
    plot_pitch(timestamps, imu0_pitch, imu1_pitch)
    plot_dorsiflexion_angle(timestamps, dorsiflexion_angles)
    # save_updated_csv(df, dorsiflexion_angles, csv_file)

if __name__ == "__main__":
    main()
