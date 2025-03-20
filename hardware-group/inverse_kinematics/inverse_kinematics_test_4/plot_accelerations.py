import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Load the CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Convert timestamps to seconds relative to the first entry
    df['imu0_timestamp'] = pd.to_datetime(df['imu0_timestamp'])
    df['imu1_timestamp'] = pd.to_datetime(df['imu1_timestamp'])
    df['imu0_time'] = (df['imu0_timestamp'] - df['imu0_timestamp'][0]).dt.total_seconds()
    df['imu1_time'] = (df['imu1_timestamp'] - df['imu1_timestamp'][0]).dt.total_seconds()
    
    return df

# Plot acceleration and angular acceleration
def plot_imu_data(df, imu_id):
    time_col = f'{imu_id}_time'
    acc_x, acc_y, acc_z = df[f'{imu_id}_acc_x'], df[f'{imu_id}_acc_y'], df[f'{imu_id}_acc_z']
    gyro_x, gyro_y, gyro_z = df[f'{imu_id}_gyro_x'], df[f'{imu_id}_gyro_y'], df[f'{imu_id}_gyro_z']
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot accelerations
    axs[0].plot(df[time_col], acc_x, label=f'{imu_id}_acc_x')
    axs[0].plot(df[time_col], acc_y, label=f'{imu_id}_acc_y')
    axs[0].plot(df[time_col], acc_z, label=f'{imu_id}_acc_z')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Acceleration (m/s²)')
    axs[0].legend()
    axs[0].set_title(f'Acceleration vs Time for {imu_id}')
    
    # Plot angular accelerations
    axs[1].plot(df[time_col], gyro_x, label=f'{imu_id}_gyro_x')
    axs[1].plot(df[time_col], gyro_y, label=f'{imu_id}_gyro_y')
    axs[1].plot(df[time_col], gyro_z, label=f'{imu_id}_gyro_z')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Angular Acceleration (rad/s²)')
    axs[1].legend()
    axs[1].set_title(f'Angular Acceleration vs Time for {imu_id}')
    
    plt.tight_layout()
    plt.show()

# Main function
def main():
    file_path = "/Users/francescobalanzoni/Documents/Python/MEng/3YP/mobisense/software-group/data-working/assets/mar12exp/2025-03-12_10-22-44-r3-limping2.csv"
    df = load_data(file_path)
    
    plot_imu_data(df, 'imu0')
    plot_imu_data(df, 'imu1')

if __name__ == "__main__":
    main()
