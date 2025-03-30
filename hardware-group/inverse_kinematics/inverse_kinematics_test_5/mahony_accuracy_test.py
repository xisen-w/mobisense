from ahrs.filters import Mahony
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.stats import pearsonr
import math

# Constants
csv_file = "/Users/francescobalanzoni/Documents/Python/MEng/3YP/mobisense/software-group/data-working/assets/mar12exp/SyncedData_OpenCapAngle_IMU/walk3_synced.csv"
frequency = 100
start_time = 0
end_time = 8.5
angle_offset = 0
time_offset = 0

kp = 0.1  # Proportional gain
ki = 0.4  # Integral gain
window_size = 50

def load_imu_data(csv_file):
    df = pd.read_csv(csv_file)
    print("Column names in the CSV:", df.columns)
    return df

def remove_drift(signal, window_size):
    return signal - pd.Series(signal).rolling(window=window_size, center=True, min_periods=1).mean()

def process_imu_data(df, kp, ki, frequency, window_size, angle_offset):
    mahony = Mahony(frequency=frequency, k_P=kp, k_I=ki)
    
    imu0_pitch = []
    imu1_pitch = []
    dorsiflexion_angles = []
    
    bias = np.zeros(3)  # Bias for accelerometer data
    q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    for i in range(len(df)):
        accel0 = np.array([df.loc[i, 'imu0_acc_x'], df.loc[i, 'imu0_acc_y'], df.loc[i, 'imu0_acc_z']])
        gyro0 = np.radians(np.array([df.loc[i, 'imu0_gyro_x'], df.loc[i, 'imu0_gyro_y'], df.loc[i, 'imu0_gyro_z']]))
        
        # Correct accelerometer data using bias
        accel0_corrected = accel0 - bias
        q0 = mahony.updateIMU(q=q0, gyr=gyro0, acc=accel0_corrected)  # Update quaternion
        r0 = R.from_quat(q0)
        _, pitch0, _ = r0.as_euler('xyz', degrees=True)
        imu0_pitch.append(pitch0)

        accel1 = np.array([df.loc[i, 'imu1_acc_x'], df.loc[i, 'imu1_acc_y'], df.loc[i, 'imu1_acc_z']])
        gyro1 = np.radians(np.array([df.loc[i, 'imu1_gyro_x'], df.loc[i, 'imu1_gyro_y'], df.loc[i, 'imu1_gyro_z']]))
        
        # Correct accelerometer data using bias
        accel1_corrected = accel1 - bias
        q1 = mahony.updateIMU(q=q1, gyr=gyro1, acc=accel1_corrected)  # Update quaternion
        r1 = R.from_quat(q1)
        _, pitch1, _ = r1.as_euler('xyz', degrees=True)
        imu1_pitch.append(pitch1)
        dorsiflexion_angles.append(pitch1 + pitch0)
        
        # Update bias correction
        bias += ki * accel0  # Apply integral correction to the bias

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
    
    absolute_errors = np.abs(df['dorsiflexion_angle'] - (df['ankle_angle_r'] + angle_offset))
    avg_absolute_error = np.mean(absolute_errors)
    
    rmse = np.sqrt(np.mean((df['dorsiflexion_angle'] - (df['ankle_angle_r'] + angle_offset)) ** 2))
    
    pearson_corr, _ = pearsonr(df['dorsiflexion_angle'], (df['ankle_angle_r'] + angle_offset))
    
    return avg_absolute_error, rmse, pearson_corr

def plot_angles(df, time_offset, start_time, end_time):
    df_filtered = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_filtered['time'], df_filtered['dorsiflexion_angle'], label='Mahony estimate', color='red', linestyle='--')
    plt.plot(df_filtered['time'] + time_offset, df_filtered['ankle_angle_r'] + angle_offset, label='OpenCap groundtruth', color='blue')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    print(f"Using constants: kp={kp}, ki={ki}, frequency={frequency}, window_size={window_size}")
    df = load_imu_data(csv_file)
    df, imu0_pitch, imu1_pitch, dorsiflexion_angles = process_imu_data(df, kp, ki, frequency, window_size, angle_offset)

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

