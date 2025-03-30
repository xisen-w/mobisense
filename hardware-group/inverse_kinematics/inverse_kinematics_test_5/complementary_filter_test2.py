from ahrs.filters import Complementary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.stats import pearsonr

# Constants
csv_file = "/Users/francescobalanzoni/Documents/Python/MEng/3YP/mobisense/software-group/data-working/assets/mar12exp/SyncedData_OpenCapAngle_IMU/walk3_synced.csv"
frequency = 100
start_time = 0
end_time = 8.5
angle_offset = 0
time_offset = 0

window_size = 50
alpha_range = np.arange(0, 0.99, 0.01)  # Testing alphas from 0.70 to 0.98

def load_imu_data(csv_file):
    df = pd.read_csv(csv_file)
    print("Column names in the CSV:", df.columns)
    return df

def remove_drift(signal, window_size):
    return signal #- pd.Series(signal).rolling(window=window_size, center=True, min_periods=1).mean()

def process_imu_data(df, alpha, frequency, window_size, angle_offset):
    complementary = Complementary(gain=alpha, frequency = frequency)
    
    imu0_pitch = []
    imu1_pitch = []
    dorsiflexion_angles = []
    
    q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    for i in range(len(df)):
        accel0 = np.array([df.loc[i, 'imu0_acc_x'], df.loc[i, 'imu0_acc_y'], df.loc[i, 'imu0_acc_z']])
        gyro0 = np.radians(np.array([df.loc[i, 'imu0_gyro_x'], df.loc[i, 'imu0_gyro_y'], df.loc[i, 'imu0_gyro_z']]))
        
        q0 = complementary.update(q0, gyr=gyro0, acc=accel0)
        r0 = R.from_quat(q0)
        _, pitch0, _ = r0.as_euler('xyz', degrees=True)
        imu0_pitch.append(pitch0)

        accel1 = np.array([df.loc[i, 'imu1_acc_x'], df.loc[i, 'imu1_acc_y'], df.loc[i, 'imu1_acc_z']])
        gyro1 = np.radians(np.array([df.loc[i, 'imu1_gyro_x'], df.loc[i, 'imu1_gyro_y'], df.loc[i, 'imu1_gyro_z']]))
        
        q1 = complementary.update(q1, gyr=gyro1, acc=accel1)
        r1 = R.from_quat(q1)
        _, pitch1, _ = r1.as_euler('xyz', degrees=True)
        imu1_pitch.append(pitch1)
        dorsiflexion_angles.append(pitch1 + pitch0)

    df['dorsiflexion_angle'] = remove_drift(dorsiflexion_angles, window_size)
    return df, imu0_pitch, imu1_pitch, dorsiflexion_angles

def compute_error_metrics(df, start_time, end_time):
    df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
    df = df.dropna(subset=['time', 'dorsiflexion_angle', 'ankle_angle_r'])
    
    absolute_errors = np.abs(df['dorsiflexion_angle'] - (df['ankle_angle_r'] + angle_offset))
    avg_absolute_error = np.mean(absolute_errors)
    rmse = np.sqrt(np.mean((df['dorsiflexion_angle'] - (df['ankle_angle_r'] + angle_offset)) ** 2))
    pearson_corr, _ = pearsonr(df['dorsiflexion_angle'], (df['ankle_angle_r'] + angle_offset))
    
    return avg_absolute_error, rmse, pearson_corr

def main():
    df_original = load_imu_data(csv_file)
    best_alpha = None
    best_rmse = float('inf')

    print("Tuning alpha value for best performance...")
    
    for alpha in alpha_range:
        df_copy = df_original.copy(deep=True)  # Ensure a fresh copy each iteration
        df_copy, _, _, _ = process_imu_data(df_copy, alpha, frequency, window_size, angle_offset)
        avg_error, rmse, pearson_corr = compute_error_metrics(df_copy, df_copy['time'].min(), df_copy['time'].max())
        
        print(f"Alpha: {alpha:.2f} -> RMSE: {rmse:.4f}, Pearson Corr: {pearson_corr:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
    
    print(f"\nBest alpha found: {best_alpha:.2f} with RMSE: {best_rmse:.4f}")
    df, imu0_pitch, imu1_pitch, dorsiflexion_angles = process_imu_data(df_original.copy(deep=True), best_alpha, frequency, window_size, angle_offset)

    avg_error, rmse, pearson_corr = compute_error_metrics(df, df['time'].min(), df['time'].max())
    
    print(f"\nFinal Performance with Alpha {best_alpha:.2f}:")
    print(f"Average Absolute Error: {avg_error:.4f} degrees")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f} degrees")
    print(f"Pearson Correlation Coefficient: {pearson_corr:.4f}")

if __name__ == "__main__":
    main()
