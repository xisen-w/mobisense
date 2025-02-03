import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, trapz, detrend
import os
from datetime import datetime

# Directory setup
current_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(current_dir, 'plots')
results_dir = os.path.join(current_dir, 'results')

# Create directories with explicit permissions
os.makedirs(plots_dir, mode=0o755, exist_ok=True)
os.makedirs(results_dir, mode=0o755, exist_ok=True)

# Constants for calculations
deg_to_rad = np.pi / 180  # Conversion factor from degrees to radians
sampling_interval = 0.01  # 10ms sampling interval

# Function to preprocess data, calculate angles, and detect peaks
def preprocess_and_analyze(data, label):
    # First, filter the gyro data to remove noise and drift
    nyquist = 100 / 2  # 100 Hz sampling rate
    low = 0.3 / nyquist
    high = 3.0 / nyquist
    b, a = butter(2, [low, high], btype='band')
    
    # Filter each axis
    filtered_gyro_x = filtfilt(b, a, data['imu0_gyro_x'])
    filtered_gyro_y = filtfilt(b, a, data['imu0_gyro_y'])
    filtered_gyro_z = filtfilt(b, a, data['imu0_gyro_z'])
    
    # Initialize angle arrays
    angles_x = np.zeros_like(filtered_gyro_x)
    angles_y = np.zeros_like(filtered_gyro_y)
    angles_z = np.zeros_like(filtered_gyro_z)
    
    # Integrate gyro data with stride reset (approximately 1-second stride)
    stride_duration = 100  # samples (at 100 Hz)
    
    for i in range(0, len(filtered_gyro_x), stride_duration):
        end_idx = min(i + stride_duration, len(filtered_gyro_x))
        # Trapezoidal integration for better accuracy
        angles_x[i:end_idx] = trapz(filtered_gyro_x[i:end_idx], dx=sampling_interval)
        angles_y[i:end_idx] = trapz(filtered_gyro_y[i:end_idx], dx=sampling_interval)
        angles_z[i:end_idx] = trapz(filtered_gyro_z[i:end_idx], dx=sampling_interval)
    
    # Convert to degrees and remove any remaining trend
    angles_x = np.rad2deg(detrend(angles_x))
    angles_y = np.rad2deg(detrend(angles_y))
    angles_z = np.rad2deg(detrend(angles_z))
    
    # Store processed angles
    data['imu0_angle_x'] = angles_x
    data['imu0_angle_y'] = angles_y
    data['imu0_angle_z'] = angles_z
    
    # Compute magnitude of acceleration for peak detection
    data['imu0_acc_magnitude'] = np.sqrt(
        data['imu0_acc_x']**2 + data['imu0_acc_y']**2 + data['imu0_acc_z']**2
    )
    
    # Detect peaks with improved parameters
    peaks, _ = find_peaks(
        data['imu0_acc_magnitude'],
        height=200,          # Minimum height threshold
        distance=50,         # Minimum samples between peaks (0.5s at 100Hz)
        prominence=50,       # Minimum prominence for peak detection
        width=1             # Minimum width of peaks
    )
    
    # Save plots with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Enhanced acceleration plot
    plt.figure(figsize=(15, 8))
    plt.plot(pd.to_datetime(data['imu0_timestamp']), data['imu0_acc_magnitude'], 
             label=f'{label} - Acc Magnitude', alpha=0.7, linewidth=1)
    plt.scatter(pd.to_datetime(data['imu0_timestamp']).iloc[peaks],
                data['imu0_acc_magnitude'].iloc[peaks], 
                color='red', label=f'Detected Steps ({len(peaks)} steps)', 
                marker='x', s=100)
    plt.title(f'Acceleration Magnitude with Detected Steps (IMU0) - {label}')
    plt.xlabel('Time')
    plt.ylabel('Acceleration Magnitude (mg)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    acc_plot_path = os.path.join(plots_dir, f'{label}_acceleration_{timestamp}.png')
    plt.savefig(acc_plot_path, dpi=300)
    plt.close()
    
    # Enhanced angles plot
    plt.figure(figsize=(15, 8))
    plt.plot(pd.to_datetime(data['imu0_timestamp']), data['imu0_angle_x'], 
             label='X (Roll)', alpha=0.7)
    plt.plot(pd.to_datetime(data['imu0_timestamp']), data['imu0_angle_y'], 
             label='Y (Pitch)', alpha=0.7)
    plt.plot(pd.to_datetime(data['imu0_timestamp']), data['imu0_angle_z'], 
             label='Z (Yaw)', alpha=0.7)
    plt.title(f'Joint Angles (IMU0) - {label}')
    plt.xlabel('Time')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    angles_plot_path = os.path.join(plots_dir, f'{label}_angles_{timestamp}.png')
    plt.savefig(angles_plot_path, dpi=300)
    plt.close()
    
    # Return enhanced results
    return {
        "Step Count": len(peaks),
        "Max Angle X (Roll)": f"{np.max(np.abs(angles_x)):.2f}°",
        "Max Angle Y (Pitch)": f"{np.max(np.abs(angles_y)):.2f}°",
        "Max Angle Z (Yaw)": f"{np.max(np.abs(angles_z)):.2f}°",
        "Average Step Interval": f"{np.mean(np.diff(peaks)) * sampling_interval:.3f} seconds",
        "Cadence": f"{60 / (np.mean(np.diff(peaks)) * sampling_interval):.1f} steps/minute"
    }

def main():
    try:
        # Load data using proper path joining
        file1_path = os.path.join(current_dir, 'assets/jan23exp/fran_walking_2025-01-22_17-21-12.csv')
        file2_path = os.path.join(current_dir, 'assets/jan23exp/xisen_walking_2025-01-22_17-46-40.csv')

        data1 = pd.read_csv(file1_path)
        data2 = pd.read_csv(file2_path)

        # Analyze both datasets
        results_fran = preprocess_and_analyze(data1, "Fran")
        results_xisen = preprocess_and_analyze(data2, "Xisen")

        # Save results to a file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join(results_dir, f'analysis_results_{timestamp}.txt')
        
        with open(results_path, 'w') as f:
            f.write("Gait Analysis Results\n==================\n\n")
            f.write(f"Fran Analysis Results:\n{results_fran}\n\n")
            f.write(f"Xisen Analysis Results:\n{results_xisen}\n")

        print(f"Results saved to: {results_path}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()