import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation as R
from scipy.stats import pearsonr

# Constants
csv_file = "/Users/francescobalanzoni/Documents/Python/MEng/3YP/mobisense/software-group/data-working/assets/mar12exp/SyncedData_OpenCapAngle_IMU/walk3_synced.csv"
frequency = 100
start_time = 0.5  
end_time = 11
angle_offset = -3
time_offset = 0

# Parameter ranges
gain_values = [0.001, 0.01, 0.1]
zeta_values = [0.01, 0.1]
window_sizes = [30, 35, 40]

class MadgwickWithBias(Madgwick):
    def __init__(self, gain, frequency, zeta):
        super().__init__(gain=gain, frequency=frequency)
        self.zeta = zeta
        self.bias = np.zeros(3)  # Bias for accelerometer data
    
    def updateIMU(self, q, gyr, acc):
        acc_corrected = acc - self.zeta * self.bias
        q[:] = super().updateIMU(q, gyr=gyr, acc=acc_corrected)
        self.bias += self.zeta * (acc - acc_corrected)
        return q

def load_imu_data(csv_file):
    return pd.read_csv(csv_file)

def remove_drift(signal, window_size):
    return signal - pd.Series(signal).rolling(window=window_size, center=True, min_periods=1).mean()

def process_imu_data(df, gain, frequency, zeta, window_size, angle_offset):
    madgwick = MadgwickWithBias(gain, frequency, zeta)
    imu0_pitch, imu1_pitch, dorsiflexion_angles = [], [], []
    q0, q1 = np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])
    
    for i in range(len(df)):
        accel0 = df.loc[i, ['imu0_acc_x', 'imu0_acc_y', 'imu0_acc_z']].values
        gyro0 = np.radians(df.loc[i, ['imu0_gyro_x', 'imu0_gyro_y', 'imu0_gyro_z']].values)
        q0 = madgwick.updateIMU(q0, gyr=gyro0, acc=accel0)
        pitch0 = R.from_quat(q0).as_euler('xyz', degrees=True)[1]
        
        accel1 = df.loc[i, ['imu1_acc_x', 'imu1_acc_y', 'imu1_acc_z']].values
        gyro1 = np.radians(df.loc[i, ['imu1_gyro_x', 'imu1_gyro_y', 'imu1_gyro_z']].values)
        q1 = madgwick.updateIMU(q1, gyr=gyro1, acc=accel1)
        pitch1 = R.from_quat(q1).as_euler('xyz', degrees=True)[1]
        
        imu0_pitch.append(pitch0)
        imu1_pitch.append(pitch1)
        dorsiflexion_angles.append(pitch1 + pitch0)
    
    df['dorsiflexion_angle'] = remove_drift(dorsiflexion_angles, window_size) + angle_offset
    return df

def compute_error_metrics(df, start_time, end_time):
    df = df[(df['time'] >= start_time) & (df['time'] <= end_time)].dropna(subset=['time', 'dorsiflexion_angle', 'ankle_angle_r'])
    absolute_errors = np.abs(df['dorsiflexion_angle'] - df['ankle_angle_r'])
    rmse = np.sqrt(np.mean((df['dorsiflexion_angle'] - df['ankle_angle_r']) ** 2))
    pearson_corr, _ = pearsonr(df['dorsiflexion_angle'], df['ankle_angle_r'])
    return np.mean(absolute_errors), rmse, pearson_corr

def optimize_parameters(csv_file):
    df = load_imu_data(csv_file)
    best_params = {}
    best_rmse = float('inf')
    
    for gain in gain_values:
        for zeta in zeta_values:
            for window_size in window_sizes:
                processed_df = process_imu_data(df.copy(), gain, frequency, zeta, window_size, angle_offset)
                avg_error, rmse, pearson_corr = compute_error_metrics(processed_df, start_time, end_time)
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {'gain': gain, 'zeta': zeta, 'window_size': window_size,
                                   'avg_error': avg_error, 'rmse': rmse, 'pearson_corr': pearson_corr}
    
    print(f"Best parameters: Gain={best_params['gain']}, Zeta={best_params['zeta']}, Window Size={best_params['window_size']}")
    print(f"Best RMSE: {best_params['rmse']:.4f} degrees")
    print(f"Average Absolute Error: {best_params['avg_error']:.4f} degrees")
    print(f"Pearson Correlation Coefficient: {best_params['pearson_corr']:.4f}")
    return best_params

if __name__ == "__main__":
    best_parameters = optimize_parameters(csv_file)
