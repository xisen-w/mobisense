import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation as R
from scipy.stats import pearsonr
import math

# Constants
csv_file = "/Users/francescobalanzoni/Documents/Python/MEng/3YP/mobisense/software-group/data-working/assets/mar12exp/SyncedData_OpenCapAngle_IMU/walk1_synced.csv"
frequency = 100
start_time = 0.5
end_time = 11
angle_offset = 0
time_offset = 0

gain = 0.001
zeta = 0.01
window_size = 50

class MadgwickWithBias(Madgwick):
    def __init__(self, gain, frequency, zeta):
        super().__init__(gain=gain, frequency=frequency, zeta=zeta)
        self.zeta = zeta
        self.bias = np.zeros(3)  # Bias for accelerometer data

    def updateIMU(self, q, gyr, acc):
        acc_corrected = acc - self.zeta * self.bias
        q[:] = super().updateIMU(q, gyr=gyr, acc=acc_corrected)
        self.bias += self.zeta * (acc - acc_corrected)
        return q

def load_imu_data(csv_file):
    df = pd.read_csv(csv_file)
    print("Column names in the CSV:", df.columns)
    return df

def remove_drift(signal, window_size):
    return signal - pd.Series(signal).rolling(window=window_size, center=True, min_periods=1).mean()

def process_imu_data(df, gain, frequency, zeta, window_size, angle_offset):
    madgwick = MadgwickWithBias(gain, frequency, zeta)
    
    imu0_pitch = []
    imu1_pitch = []
    dorsiflexion_angles = []

    q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q1 = np.array([1, 0.0, 0, 0.0], dtype=np.float64)

    for i in range(len(df)):
        accel0 = np.array([df.loc[i, 'imu0_acc_x'], df.loc[i, 'imu0_acc_y'], df.loc[i, 'imu0_acc_z']])
        gyro0 = np.radians(np.array([df.loc[i, 'imu0_gyro_x'], df.loc[i, 'imu0_gyro_y'], df.loc[i, 'imu0_gyro_z']]))
        q0 = madgwick.updateIMU(q0, gyr=gyro0, acc=accel0)
        r0 = R.from_quat(q0)
        _, pitch0, _ = r0.as_euler('xyz', degrees=True)
        imu0_pitch.append(pitch0)

        accel1 = np.array([df.loc[i, 'imu1_acc_x'], df.loc[i, 'imu1_acc_y'], df.loc[i, 'imu1_acc_z']])
        gyro1 = np.radians(np.array([df.loc[i, 'imu1_gyro_x'], df.loc[i, 'imu1_gyro_y'], df.loc[i, 'imu1_gyro_z']]))
        q1 = madgwick.updateIMU(q1, gyr=gyro1, acc=accel1)
        r1 = R.from_quat(q1)
        _, pitch1, _ = r1.as_euler('xyz', degrees=True)
        imu1_pitch.append(pitch1)
        dorsiflexion_angles.append(pitch1 + pitch0)
    
    df['dorsiflexion_angle'] = remove_drift(dorsiflexion_angles, window_size)
    return df, imu0_pitch, imu1_pitch, dorsiflexion_angles

def compute_error_metrics(df, start_time, end_time):
    if 'time' not in df.columns:
        print("Error: 'time' column is missing.")
        return None, None, None
    df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
    if 'dorsiflexion_angle' not in df.columns:
        print("Error: dorsiflexion_angle column is missing.")
        return None, None, None
    df = df.dropna(subset=['time', 'dorsiflexion_angle', 'ankle_angle_r'])
    
    absolute_errors = np.abs(df['dorsiflexion_angle'] - (df['ankle_angle_r']+angle_offset))
    avg_absolute_error = np.mean(absolute_errors)
    
    rmse = np.sqrt(np.mean((df['dorsiflexion_angle'] - (df['ankle_angle_r']+angle_offset)) ** 2))
    
    pearson_corr, _ = pearsonr(df['dorsiflexion_angle'], (df['ankle_angle_r']+angle_offset))
    
    return avg_absolute_error, rmse, pearson_corr

def plot_angles(df, time_offset, start_time, end_time):
    df_filtered = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
    
    # Define standard deviation range (±5 degrees around ground truth)
    ground_truth = df_filtered['ankle_angle_r'] + angle_offset
    std_dev_range_upper = ground_truth + 5
    std_dev_range_lower = ground_truth - 5
    
    plt.figure(figsize=(10, 6))
    
    # Plot the angles
    plt.plot(df_filtered['time'], df_filtered['dorsiflexion_angle'], label='Madgwick estimate', color='red', linestyle='--')
    plt.plot(df_filtered['time'] + time_offset, ground_truth, label='OpenCap groundtruth', color='blue')
    
    # Fill the standard deviation range
    plt.fill_between(df_filtered['time'] + time_offset, std_dev_range_lower, std_dev_range_upper, color='blue', alpha=0.2, label='Ground truth ±5°')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    print(f"Using constants: gain={gain}, frequency={frequency}, zeta={zeta}, window_size={window_size}")
    df = load_imu_data(csv_file)
    df, imu0_pitch, imu1_pitch, dorsiflexion_angles = process_imu_data(df, gain, frequency, zeta, window_size, angle_offset)

    start_time = df['time'].min()
    end_time = df['time'].max()

    plot_angles(df, time_offset, start_time, end_time)
    avg_error, rmse, pearson_corr = compute_error_metrics(df, start_time, end_time)
    
    if avg_error is not None:
        print(f"Average Absolute Error: {avg_error:.4f} degrees")
        print(f"Root Mean Square Error (RMSE): {rmse:.4f} degrees")
        print(f"Pearson Correlation Coefficient: {pearson_corr:.4f}")
    else:
        print("Error in computing error metrics.")

if __name__ == "__main__":
    main()
