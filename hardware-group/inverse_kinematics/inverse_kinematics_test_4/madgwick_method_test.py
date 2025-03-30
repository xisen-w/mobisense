import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
from datetime import datetime

"""
def butter_lowpass_filter(data, cutoff=5, fs=27, order=3):
    # Applies a low-pass Butterworth filter to the given data.
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

"""
def butter_highpass_filter(data, cutoff=0.7, fs=29, order=3):
    # Applies a high-pass Butterworth filter to remove drift in gyro data.
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=0)

def load_imu_data(csv_file):
    """Loads IMU data from a CSV file."""
    df = pd.read_csv(csv_file)
    return df

def process_imu_data(df):
    """Processes IMU data using Madgwick filter with low-pass and high-pass filtering."""
    madgwick = Madgwick(beta=0.001)
    
    imu0_pitch = []
    imu1_pitch = []
    timestamps = []

    q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    accel_data_0 = df[['imu0_acc_x', 'imu0_acc_y', 'imu0_acc_z']].values
    gyro_data_0 = np.radians(df[['imu0_gyro_x', 'imu0_gyro_y', 'imu0_gyro_z']].values)
    
    accel_data_1 = df[['imu1_acc_x', 'imu1_acc_y', 'imu1_acc_z']].values
    gyro_data_1 = np.radians(df[['imu1_gyro_x', 'imu1_gyro_y', 'imu1_gyro_z']].values)
    
    # Apply filters
    #accel_data_0 = butter_lowpass_filter(accel_data_0)
    gyro_data_0 = butter_highpass_filter(gyro_data_0)

    #accel_data_1 = butter_lowpass_filter(accel_data_1)
    gyro_data_1 = butter_highpass_filter(gyro_data_1)

    for i in range(len(df)):
        accel0 = accel_data_0[i]
        gyro0 = gyro_data_0[i]
        accel1 = accel_data_1[i]
        gyro1 = gyro_data_1[i]
        
        # Compute orientation using Madgwick
        q0 = madgwick.updateIMU(q0, gyr=gyro0, acc=accel0)
        r0 = R.from_quat(q0)
        _, pitch0, _ = r0.as_euler('xyz', degrees=True)
        imu0_pitch.append(pitch0)

        q1 = madgwick.updateIMU(q1, gyr=gyro1, acc=accel1)
        r1 = R.from_quat(q1)
        _, pitch1, _ = r1.as_euler('xyz', degrees=True)
        imu1_pitch.append(pitch1)

        # Convert timestamp to datetime format
        timestamps.append(datetime.fromisoformat(df.loc[i, 'imu0_timestamp']))
    
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
    pitch_diff = [p1 + p0 for p0, p1 in zip(imu0_pitch, imu1_pitch)]
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, pitch_diff)
    plt.xlabel("Time")
    plt.ylabel("Plantar Flexion Angle (degrees)")
    plt.grid()
    plt.xticks(rotation=45)
    plt.show()

def main():
    csv_file = "/Users/francescobalanzoni/Documents/Python/MEng/3YP/mobisense/software-group/data-working/assets/mar12exp/2025-03-12_10-20-54-r2-limping1.csv"
    df = load_imu_data(csv_file)
    
    timestamps, imu0_pitch, imu1_pitch = process_imu_data(df)
    plot_pitch(timestamps, imu0_pitch, imu1_pitch)
    plot_pitch_difference(timestamps, imu0_pitch, imu1_pitch)

if __name__ == "__main__":
    main()
