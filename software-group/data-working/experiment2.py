import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path
import os

# Create directories if they don't exist
plots_dir = Path('./plots')
results_dir = Path('./results')
plots_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

def load_and_preprocess_data(file_path):
    """Load and preprocess the IMU data."""
    data = pd.read_csv(file_path)
    
    # Convert timestamps to datetime
    data['imu0_timestamp'] = pd.to_datetime(data['imu0_timestamp'])
    data['imu1_timestamp'] = pd.to_datetime(data['imu1_timestamp'])
    
    # Calculate acceleration magnitudes and smooth
    for imu in ['imu0', 'imu1']:
        data[f'{imu}_acc_magnitude'] = np.sqrt(
            data[f'{imu}_acc_x']**2 + 
            data[f'{imu}_acc_y']**2 + 
            data[f'{imu}_acc_z']**2
        )
        # Convert to m/s² from mg
        data[f'{imu}_acc_magnitude_mps2'] = (data[f'{imu}_acc_magnitude'] / 1000) * 9.81
        # Smooth the data
        data[f'{imu}_acc_magnitude_smooth'] = data[f'{imu}_acc_magnitude_mps2'].rolling(
            window=5, center=True).mean()
    
    return data

def calculate_joint_angles(data):
    """Calculate joint angles using gyroscope and accelerometer data."""
    for imu in ['imu0', 'imu1']:
        # Gyroscope-based angle
        data[f'{imu}_gyro_angle'] = data[f'{imu}_gyro_x'].cumsum() * (1/100)  # Assuming 100Hz sampling
        
        # Accelerometer-based angle (example using trigonometry for dorsiflexion)
        data[f'{imu}_acc_angle'] = np.arctan2(
            data[f'{imu}_acc_y'], 
            data[f'{imu}_acc_z']
        ) * (180 / np.pi)  # Convert to degrees
        
        # Sensor fusion
        data[f'{imu}_fused_angle'] = 0.5 * data[f'{imu}_acc_angle'] + 0.5 * data[f'{imu}_gyro_angle']  # Lambda = 0.5
        
    return data

def calculate_rom_and_symmetry(data_imu0, data_imu1):
    """Calculate Range of Motion (RoM) and symmetry index."""
    rom_imu0 = data_imu0['imu0_fused_angle'].max() - data_imu0['imu0_fused_angle'].min()
    rom_imu1 = data_imu1['imu1_fused_angle'].max() - data_imu1['imu1_fused_angle'].min()
    
    # Calculate symmetry index
    asymmetry_index = abs(rom_imu0 - rom_imu1) / max(rom_imu0, rom_imu1) * 100
    return rom_imu0, rom_imu1, asymmetry_index

def perform_gait_analysis(data, subject_name):
    """Perform gait analysis and generate plots."""
    results = {}
    
    for imu in ['imu0', 'imu1']:
        # Detect peaks for step detection
        peaks, _ = find_peaks(
            data[f'{imu}_acc_magnitude_smooth'],
            height=1.5,
            distance=10,
            prominence=0.5
        )
        
        # Calculate step metrics
        step_times = np.diff(data[f'{imu}_timestamp'].iloc[peaks])
        cadence = len(peaks) / (data[f'{imu}_timestamp'].max() - 
                               data[f'{imu}_timestamp'].min()).total_seconds() * 60
        
        results[f'{imu}_peaks'] = peaks
        results[f'{imu}_cadence'] = cadence
        results[f'{imu}_avg_step_time'] = np.mean(step_times) / np.timedelta64(1, 's')
        
        # Plot acceleration data and detected steps
        plt.figure(figsize=(15, 6))
        plt.plot(data[f'{imu}_timestamp'], 
                data[f'{imu}_acc_magnitude_smooth'], 
                label='Smoothed Acceleration')
        plt.scatter(data[f'{imu}_timestamp'].iloc[peaks],
                   data[f'{imu}_acc_magnitude_smooth'].iloc[peaks],
                   color='red', label='Detected Steps')
        plt.title(f'{subject_name} - {imu.upper()} Gait Analysis')
        plt.xlabel('Time')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / f'{subject_name}_{imu}_gait_analysis.png')
        plt.close()
    
    return results

def generate_report(subject_results, rom_results):
    """Generate and save a detailed analysis report."""
    report = ["Gait Analysis Report\n===================\n\n"]
    
    for subject, results in subject_results.items():
        report.append(f"\n{subject}'s Results:\n{'-' * 20}")
        for imu in ['imu0', 'imu1']:
            report.append(f"\n{imu.upper()}:")
            report.append(f"- Number of steps: {len(results[f'{imu}_peaks'])}")
            report.append(f"- Cadence: {results[f'{imu}_cadence']:.2f} steps/minute")
            report.append(f"- Average step time: {results[f'{imu}_avg_step_time']:.2f} seconds")
        
        rom_imu0, rom_imu1, asymmetry_index = rom_results[subject]
        report.append(f"\nRange of Motion (RoM):")
        report.append(f"- IMU0: {rom_imu0:.2f}°")
        report.append(f"- IMU1: {rom_imu1:.2f}°")
        report.append(f"- Asymmetry Index: {asymmetry_index:.2f}%\n")
    
    with open(results_dir / 'gait_analysis_report.txt', 'w') as f:
        f.write('\n'.join(report))

def main():
    # Load and preprocess data
    fran_data = load_and_preprocess_data('./data/fran_walking.csv')
    xisen_data = load_and_preprocess_data('./data/xisen_walking.csv')
    
    # Calculate joint angles
    fran_data = calculate_joint_angles(fran_data)
    xisen_data = calculate_joint_angles(xisen_data)
    
    # Perform gait analysis
    results_fran = perform_gait_analysis(fran_data, 'Fran')
    results_xisen = perform_gait_analysis(xisen_data, 'Xisen')
    
    # Calculate RoM and symmetry
    rom_fran = calculate_rom_and_symmetry(fran_data, fran_data)
    rom_xisen = calculate_rom_and_symmetry(xisen_data, xisen_data)
    
    # Generate report
    generate_report(
        subject_results={'Fran': results_fran, 'Xisen': results_xisen},
        rom_results={'Fran': rom_fran, 'Xisen': rom_xisen}
    )

if __name__ == "__main__":
    main()