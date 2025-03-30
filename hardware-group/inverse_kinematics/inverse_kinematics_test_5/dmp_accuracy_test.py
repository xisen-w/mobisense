import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import math

# Constants
csv_file = "/Users/francescobalanzoni/Documents/Python/MEng/3YP/mobisense/software-group/data-working/assets/mar12exp/SyncedData_OpenCapAngle_IMU/walk3_synced.csv"
frequency = 100
start_time = 2.7
end_time = 7.6
angle_offset = -5
time_offset = -0.7

gain = 0.001
zeta = 0.01
window_size = 30

def load_imu_data(csv_file):
    df = pd.read_csv(csv_file)
    print("Column names in the CSV:", df.columns)
    return df

def remove_drift(signal, window_size):
    return signal - pd.Series(signal).rolling(window=window_size, center=True, min_periods=1).mean()

def process_imu_data(df, window_size, angle_offset):
    # Now we directly use imu0_roll and imu1_roll from the CSV
    imu0_roll = []
    imu1_roll = []
    dorsiflexion_angles = []
    
    for i in range(len(df)):
        # Directly append the roll angles from imu0 and imu1
        roll0 = df.loc[i, 'imu0_roll']
        roll1 = df.loc[i, 'imu1_roll']
        
        imu0_roll.append(roll0)
        imu1_roll.append(roll1)
        
        # Compute dorsiflexion angle as the sum of imu0_roll and imu1_roll
        dorsiflexion_angles.append(roll0 + roll1)
    
    # Apply drift removal on the dorsiflexion angles
    df['dorsiflexion_angle'] = remove_drift(dorsiflexion_angles, window_size)
    return df, imu0_roll, imu1_roll, dorsiflexion_angles

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
    plt.plot(df_filtered['time'], df_filtered['dorsiflexion_angle'], label='Calculated dorsiflexion angle', color='red', linestyle='--')
    plt.plot(df_filtered['time'] + time_offset, df_filtered['ankle_angle_r'] + angle_offset, label='OpenCap groundtruth', color='blue')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    print(f"Using constants: gain={gain}, frequency={frequency}, zeta={zeta}, window_size={window_size}")
    df = load_imu_data(csv_file)
    df, imu0_roll, imu1_roll, dorsiflexion_angles = process_imu_data(df, window_size, angle_offset)

    # start_time = df['time'].min()
    # end_time = df['time'].max()

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
