import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import find_peaks
import os
from datetime import datetime

# Directory setup
current_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(current_dir, 'plots')
results_dir = os.path.join(current_dir, 'results')

# Create directories with explicit permissions
os.makedirs(plots_dir, mode=0o755, exist_ok=True)
os.makedirs(results_dir, mode=0o755, exist_ok=True)

def calculate_joint_angles(data):
    """Calculate joint angles using gyroscope and accelerometer data."""
    sampling_freq = 100  # Hz
    dt = 1/sampling_freq
    alpha = 0.98  # Complementary filter coefficient
    
    for imu in ['imu0', 'imu1']:
        # Gyroscope-based angle (integration of angular velocity)
        data[f'{imu}_gyro_angle'] = np.cumsum(data[f'{imu}_gyro_x'] * dt)
        
        # Accelerometer-based angle
        data[f'{imu}_acc_angle'] = np.arctan2(
            data[f'{imu}_acc_y'],
            np.sqrt(data[f'{imu}_acc_x']**2 + data[f'{imu}_acc_z']**2)
        ) * (180 / np.pi)
        
        # Complementary filter
        data[f'{imu}_fused_angle'] = pd.Series(0.0, index=data.index)
        data.loc[0, f'{imu}_fused_angle'] = data.loc[0, f'{imu}_acc_angle']
        
        for i in range(1, len(data)):
            prev_angle = data.loc[i-1, f'{imu}_fused_angle']
            gyro_angle = prev_angle + data.loc[i, f'{imu}_gyro_x'] * dt
            acc_angle = data.loc[i, f'{imu}_acc_angle']
            data.loc[i, f'{imu}_fused_angle'] = alpha * gyro_angle + (1 - alpha) * acc_angle
    
    return data

def calculate_rom_and_symmetry(data):
    """Calculate Range of Motion (RoM) and symmetry index."""
    results = {}
    
    for imu in ['imu0', 'imu1']:
        # Calculate RoM
        angle_data = data[f'{imu}_fused_angle']
        rom = angle_data.max() - angle_data.min()
        results[f'{imu}_rom'] = rom
        
        # Calculate dorsiflexion and plantarflexion
        neutral_angle = np.median(angle_data)
        results[f'{imu}_dorsiflexion'] = angle_data.max() - neutral_angle
        results[f'{imu}_plantarflexion'] = neutral_angle - angle_data.min()
    
    # Calculate symmetry index
    si = abs(results['imu0_rom'] - results['imu1_rom']) / (0.5 * (results['imu0_rom'] + results['imu1_rom'])) * 100
    results['symmetry_index'] = si
    
    return results

def load_and_preprocess_data(file_path):
    """Load and preprocess the IMU data."""
    try:
        data = pd.read_csv(file_path)
        
        # Convert timestamps to datetime
        data['imu0_timestamp'] = pd.to_datetime(data['imu0_timestamp'])
        data['imu1_timestamp'] = pd.to_datetime(data['imu1_timestamp'])
        
        # Calculate acceleration magnitudes
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
        
        # Calculate joint angles
        data = calculate_joint_angles(data)
        
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def perform_gait_analysis(data, subject_name):
    """Perform gait analysis and generate plots."""
    if data is None:
        return None
    
    results = {}
    
    try:
        # Calculate RoM and symmetry
        rom_results = calculate_rom_and_symmetry(data)
        results.update(rom_results)
        
        for imu in ['imu0', 'imu1']:
            # Find peaks for step detection
            peaks, properties = find_peaks(
                data[f'{imu}_acc_magnitude_smooth'].fillna(0),
                height=1.5,
                distance=10,
                prominence=0.5
            )
            
            # Calculate gait metrics
            step_times = np.diff(data[f'{imu}_timestamp'].iloc[peaks])
            duration = (data[f'{imu}_timestamp'].max() - 
                       data[f'{imu}_timestamp'].min()).total_seconds()
            cadence = len(peaks) / duration * 60 if duration > 0 else 0
            
            results[f'{imu}_peaks'] = peaks
            results[f'{imu}_cadence'] = cadence
            results[f'{imu}_avg_step_time'] = np.mean(step_times) / np.timedelta64(1, 's')
            
            # Generate plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            
            # Plot 1: Acceleration and steps
            ax1.plot(data[f'{imu}_timestamp'], 
                    data[f'{imu}_acc_magnitude_smooth'], 
                    label='Smoothed Acceleration')
            ax1.scatter(data[f'{imu}_timestamp'].iloc[peaks],
                       data[f'{imu}_acc_magnitude_smooth'].iloc[peaks],
                       color='red', label='Detected Steps')
            ax1.set_title(f'{subject_name} - {imu.upper()} Gait Analysis')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Acceleration (m/s²)')
            ax1.legend()
            ax1.grid(True)
            
            # Plot 2: Joint angles
            ax2.plot(data[f'{imu}_timestamp'], 
                    data[f'{imu}_fused_angle'],
                    label='Joint Angle')
            ax2.set_title(f'{subject_name} - {imu.upper()} Joint Angle')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Angle (degrees)')
            ax2.legend()
            ax2.grid(True)
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(plots_dir, f'{subject_name}_{imu}_analysis_{timestamp}.png')
            plt.savefig(plot_path)
            plt.close()
            
        return results
    except Exception as e:
        print(f"Error in gait analysis: {str(e)}")
        return None

def generate_report(results_fran, results_xisen):
    """Generate and save analysis report."""
    if results_fran is None or results_xisen is None:
        print("Cannot generate report: missing analysis results")
        return
    
    try:
        report = ["Gait Analysis Report\n===================\n\n"]
        
        for subject, results in [("Fran", results_fran), ("Xisen", results_xisen)]:
            report.append(f"\n{subject}'s Results:\n{'-' * 20}")
            for imu in ['imu0', 'imu1']:
                report.append(f"\n{imu.upper()}:")
                report.append(f"- Number of steps: {len(results[f'{imu}_peaks'])}")
                report.append(f"- Cadence: {results[f'{imu}_cadence']:.2f} steps/minute")
                report.append(f"- Average step time: {results[f'{imu}_avg_step_time']:.2f} seconds")
                report.append(f"- Range of Motion: {results[f'{imu}_rom']:.2f}°")
                report.append(f"- Dorsiflexion: {results[f'{imu}_dorsiflexion']:.2f}°")
                report.append(f"- Plantarflexion: {results[f'{imu}_plantarflexion']:.2f}°")
            
            report.append(f"\nSymmetry Index: {results['symmetry_index']:.2f}%")
        
        # Save report with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(results_dir, f'gait_analysis_report_{timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
    except Exception as e:
        print(f"Error generating report: {str(e)}")

def main():
    try:
        # Load data
        fran_file = os.path.join(current_dir, 'assets/jan23exp/fran_walking_2025-01-22_17-21-12.csv')
        xisen_file = os.path.join(current_dir, 'assets/jan23exp/xisen_walking_2025-01-22_17-46-40.csv')
        
        fran_data = load_and_preprocess_data(fran_file)
        xisen_data = load_and_preprocess_data(xisen_file)
        
        # Perform analysis
        results_fran = perform_gait_analysis(fran_data, 'Fran')
        results_xisen = perform_gait_analysis(xisen_data, 'Xisen')
        
        # Generate report
        generate_report(results_fran, results_xisen)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()