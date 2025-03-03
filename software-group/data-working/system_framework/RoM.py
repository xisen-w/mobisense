import numpy as np
import pandas as pd
from scipy import signal
from typing import Dict, Optional, Tuple

class RangeOfMotionAnalyzer:
    def __init__(self, sampling_rate: float = 100.0):
        """
        Initialize the ROM analyzer
        
        Args:
            sampling_rate: Sampling frequency of the IMU data in Hz
        """
        self.sampling_rate = sampling_rate
        
    def calculate_ankle_angles(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate ankle angles from IMU data using both foot and shank IMUs
        
        Args:
            data: DataFrame with columns for both IMUs:
                  imu0_acc_x/y/z (foot), imu1_acc_x/y/z (shank)
                  imu0_gyro_x/y/z (foot), imu1_gyro_x/y/z (shank)
            
        Returns:
            Dictionary containing relative ankle angles and metrics
        """
        # Initialize results
        angles = pd.DataFrame(index=data.index)
        
        # Calculate angles for both IMUs
        # Foot IMU (IMU0) angles
        foot_pitch = np.arctan2(-data['imu0_acc_x'], 
                               np.sqrt(data['imu0_acc_y']**2 + data['imu0_acc_z']**2))
        foot_roll = np.arctan2(data['imu0_acc_y'], 
                              np.sqrt(data['imu0_acc_x']**2 + data['imu0_acc_z']**2))
        
        # Shank IMU (IMU1) angles
        shank_pitch = np.arctan2(-data['imu1_acc_x'], 
                                np.sqrt(data['imu1_acc_y']**2 + data['imu1_acc_z']**2))
        shank_roll = np.arctan2(data['imu1_acc_y'], 
                               np.sqrt(data['imu1_acc_x']**2 + data['imu1_acc_z']**2))
        
        # Calculate relative ankle angles
        # Note: The order of subtraction matters for anatomical meaning
        angles['sagittal_angle'] = (foot_pitch - shank_pitch) * (180 / np.pi)  # Dorsiflexion(+)/Plantarflexion(-)
        angles['frontal_angle'] = (foot_roll - shank_roll) * (180 / np.pi)     # Inversion(+)/Eversion(-)
        
        # Add complementary filter for gyro integration
        dt = 1/self.sampling_rate
        alpha = 0.96  # Filter coefficient
        
        # Initialize integrated angles
        gyro_sagittal = np.zeros_like(foot_pitch)
        gyro_frontal = np.zeros_like(foot_roll)
        
        # Integrate gyroscope data (relative angular velocity between foot and shank)
        for i in range(1, len(data)):
            # Sagittal plane (around y-axis)
            gyro_sagittal[i] = gyro_sagittal[i-1] + \
                              (data['imu0_gyro_y'].iloc[i] - data['imu1_gyro_y'].iloc[i]) * dt
            
            # Frontal plane (around x-axis)
            gyro_frontal[i] = gyro_frontal[i-1] + \
                             (data['imu0_gyro_x'].iloc[i] - data['imu1_gyro_x'].iloc[i]) * dt
        
        # Complementary filter
        angles['sagittal_angle'] = alpha * gyro_sagittal + (1 - alpha) * angles['sagittal_angle']
        angles['frontal_angle'] = alpha * gyro_frontal + (1 - alpha) * angles['frontal_angle']
        
        return {
            'angles': angles,
            'metrics': self.calculate_rom_metrics(angles)
        }
    
    def calculate_rom_metrics(self, angles: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate ROM metrics from angle data
        
        Args:
            angles: DataFrame with calculated angles
            
        Returns:
            Dictionary containing ROM metrics
        """
        metrics = {}
        
        # Sagittal plane (dorsiflexion/plantarflexion)
        sagittal = angles['sagittal_angle']
        neutral_sagittal = np.median(sagittal)
        
        metrics.update({
            'dorsiflexion_max': float(sagittal.max() - neutral_sagittal),
            'plantarflexion_max': float(neutral_sagittal - sagittal.min()),
            'sagittal_rom': float(sagittal.max() - sagittal.min()),
            'neutral_sagittal': float(neutral_sagittal)
        })
        
        # Frontal plane (inversion/eversion)
        frontal = angles['frontal_angle']
        neutral_frontal = np.median(frontal)
        
        metrics.update({
            'inversion_max': float(frontal.max() - neutral_frontal),
            'eversion_max': float(neutral_frontal - frontal.min()),
            'frontal_rom': float(frontal.max() - frontal.min()),
            'neutral_frontal': float(neutral_frontal)
        })
        
        return metrics
    
    def calculate_sprain_risk_metrics(self, angles: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate additional metrics relevant for ankle sprain assessment
        
        Args:
            angles: DataFrame with calculated angles
            
        Returns:
            Dictionary containing sprain risk metrics
        """
        metrics = {}
        
        # Calculate rate of angle change
        sagittal_velocity = np.gradient(angles['sagittal_angle'].values, 1/self.sampling_rate)
        frontal_velocity = np.gradient(angles['frontal_angle'].values, 1/self.sampling_rate)
        
        # Calculate metrics relevant to sprain risk
        metrics.update({
            'max_inversion_velocity': float(np.max(np.abs(frontal_velocity))),
            'sudden_inversion_count': len(np.where(frontal_velocity > 200)[0]),  # Threshold for sudden inversion
            'max_combined_velocity': float(np.max(np.sqrt(sagittal_velocity**2 + frontal_velocity**2))),
            'stability_index': float(np.std(frontal_velocity))  # Higher values indicate less stability
        })
        
        return metrics
    
    def analyze_dynamic_rom(self, data: pd.DataFrame, 
                          window_size: float = 1.0) -> Dict[str, pd.DataFrame]:
        """
        Analyze ROM during dynamic movements
        
        Args:
            data: IMU data DataFrame
            window_size: Analysis window size in seconds
            
        Returns:
            Dictionary containing dynamic ROM analysis
        """
        window_samples = int(window_size * self.sampling_rate)
        
        # Calculate angles for the entire dataset
        angles_dict = self.calculate_ankle_angles(data)
        angles = angles_dict['angles']
        i
        # Initialize dynamic metrics
        dynamic_metrics = pd.DataFrame()
        
        for start in range(0, len(data) - window_samples, window_samples // 2):
            end = start + window_samples
            window_angles = angles.iloc[start:end]
            
            metrics = self.calculate_rom_metrics(window_angles)
            metrics['start_time'] = start / self.sampling_rate
            metrics['end_time'] = end / self.sampling_rate
            
            dynamic_metrics = pd.concat([
                dynamic_metrics, 
                pd.DataFrame([metrics])
            ])
        
        return {
            'angles': angles,
            'dynamic_metrics': dynamic_metrics.reset_index(drop=True)
        }

if __name__ == "__main__":
    # Load test data using absolute path
    data_path = "/Users/wangxiang/Desktop/my_workspace/mobisense/software-group/data-working/assets/feb16exp/2025-02-16_18-15-14_round_1.csv"
    data = pd.read_csv(data_path)
    
    # Initialize analyzer
    rom_analyzer = RangeOfMotionAnalyzer()
    
    # Calculate angles and metrics
    results = rom_analyzer.calculate_ankle_angles(data)
    sprain_risk = rom_analyzer.calculate_sprain_risk_metrics(results['angles'])
    
    # Print results
    print("\nRange of Motion Metrics:")
    print("-----------------------")
    for key, value in results['metrics'].items():
        print(f"{key}: {value:.2f}°")
    
    print("\nSprain Risk Metrics:")
    print("------------------")
    for key, value in sprain_risk.items():
        print(f"{key}: {value:.2f}")
    
    # Plot results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(results['angles']['sagittal_angle'], label='Dorsiflexion/Plantarflexion')
    plt.plot(results['angles']['frontal_angle'], label='Inversion/Eversion')
    plt.title('Ankle Angles Over Time')
    plt.xlabel('Samples')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid(True)
    plt.show()
