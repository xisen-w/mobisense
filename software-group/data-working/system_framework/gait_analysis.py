import numpy as np
import pandas as pd
from scipy import signal
from typing import Dict, List, Optional, Tuple
from scipy.signal import find_peaks

class GaitAnalyzer:
    def __init__(self, sampling_rate: float = 100.0):
        """
        Initialize the gait analyzer
        
        Args:
            sampling_rate: Sampling frequency of the IMU data in Hz
        """
        self.sampling_rate = sampling_rate
        
    def detect_gait_events(self, shank_acc: np.ndarray, 
                          foot_acc: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect gait events using both shank and foot IMU data
        
        Args:
            shank_acc: Acceleration data from shank IMU
            foot_acc: Acceleration data from foot IMU
            
        Returns:
            Dictionary containing indices of heel strike and toe-off events
        """
        # Calculate magnitudes
        shank_mag = np.linalg.norm(shank_acc, axis=1)
        foot_mag = np.linalg.norm(foot_acc, axis=1)
        
        # Filter signals
        b, a = signal.butter(4, [0.5, 3.0], btype='band', fs=self.sampling_rate)
        filtered_shank = signal.filtfilt(b, a, shank_mag)
        filtered_foot = signal.filtfilt(b, a, foot_mag)
        
        # Detect heel strikes using shank IMU
        heel_strikes, _ = find_peaks(filtered_shank,
                                   height=np.mean(filtered_shank),
                                   distance=int(0.5 * self.sampling_rate),
                                   prominence=0.5)
        
        # Detect toe-offs using foot IMU
        toe_offs, _ = find_peaks(filtered_foot,
                                height=np.mean(filtered_foot),
                                distance=int(0.5 * self.sampling_rate),
                                prominence=0.5)
        
        return {
            'heel_strikes': heel_strikes,
            'toe_offs': toe_offs
        }
    
    def calculate_gait_parameters(self, events: Dict[str, np.ndarray], 
                                angles: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate gait parameters using detected events and ankle angles
        
        Args:
            events: Dictionary with heel strike and toe-off indices
            angles: DataFrame with ankle angles
            
        Returns:
            Dictionary containing gait parameters
        """
        heel_strikes = events['heel_strikes']
        toe_offs = events['toe_offs']
        
        # Calculate temporal parameters
        stride_times = np.diff(heel_strikes) / self.sampling_rate
        stance_phases = []
        
        for hs in heel_strikes:
            # Find next toe-off
            next_to = toe_offs[toe_offs > hs][0] if len(toe_offs[toe_offs > hs]) > 0 else None
            if next_to is not None:
                stance_phases.append((next_to - hs) / self.sampling_rate)
        
        # Calculate metrics
        metrics = {
            'stride_count': len(heel_strikes) - 1,
            'cadence': len(heel_strikes) / (len(angles) / self.sampling_rate) * 60,
            'mean_stride_time': float(np.mean(stride_times)) if len(stride_times) > 0 else 0,
            'stance_phase_ratio': float(np.mean(stance_phases) / np.mean(stride_times)) if len(stride_times) > 0 else 0,
            'stride_time_variability': float(np.std(stride_times)) if len(stride_times) > 0 else 0
        }
        
        return metrics
    
    def analyze_pathological_gait(self, angles: pd.DataFrame, 
                                events: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze gait patterns for potential pathological indicators
        
        Args:
            angles: DataFrame with ankle angles
            events: Dictionary with gait events
            
        Returns:
            Dictionary containing pathological gait metrics
        """
        sagittal = angles['sagittal_angle'].values
        frontal = angles['frontal_angle'].values
        
        # Calculate metrics specific to ankle instability
        metrics = {
            'peak_inversion_during_stance': [],
            'inversion_variability': [],
            'early_heel_rise': 0
        }
        
        for i, hs in enumerate(events['heel_strikes'][:-1]):
            next_to = events['toe_offs'][events['toe_offs'] > hs][0]
            stance_phase = frontal[hs:next_to]
            
            metrics['peak_inversion_during_stance'].append(np.max(stance_phase))
            metrics['inversion_variability'].append(np.std(stance_phase))
            
            # Check for early heel rise
            if np.mean(sagittal[hs:next_to]) > 15:  # Threshold for early heel rise
                metrics['early_heel_rise'] += 1
        
        # Summarize metrics
        return {
            'mean_peak_inversion': float(np.mean(metrics['peak_inversion_during_stance'])),
            'inversion_variability': float(np.mean(metrics['inversion_variability'])),
            'early_heel_rise_ratio': float(metrics['early_heel_rise'] / len(events['heel_strikes']))
        }
    
    def calculate_stride_parameters(self, acc_data: np.ndarray, 
                                 step_indices: np.ndarray) -> Dict[str, float]:
        """
        Calculate stride parameters from acceleration data and detected steps
        
        Args:
            acc_data: Acceleration data
            step_indices: Indices of detected steps
            
        Returns:
            Dictionary containing stride parameters
        """
        if len(step_indices) < 2:
            return {
                'stride_length': 0,
                'stride_time': 0,
                'stride_velocity': 0
            }
        
        # Calculate stride time (two steps)
        stride_times = np.diff(step_indices[::2]) / self.sampling_rate
        
        # Estimate stride length using double integration
        stride_lengths = []
        for i in range(0, len(step_indices)-2, 2):
            start_idx = step_indices[i]
            end_idx = step_indices[i+2]
            
            # First integration for velocity
            velocity = np.cumsum(acc_data[start_idx:end_idx]) / self.sampling_rate
            
            # Second integration for position
            position = np.cumsum(velocity) / self.sampling_rate
            
            # Estimate stride length
            stride_lengths.append(np.max(np.abs(position)))
        
        # Calculate metrics
        mean_stride_time = np.mean(stride_times)
        mean_stride_length = np.mean(stride_lengths) if stride_lengths else 0
        
        return {
            'stride_length': mean_stride_length,
            'stride_time': mean_stride_time,
            'stride_velocity': mean_stride_length / mean_stride_time if mean_stride_time > 0 else 0
        }
    
    def analyze_gait_symmetry(self, left_data: pd.DataFrame, 
                            right_data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze gait symmetry between left and right sides
        
        Args:
            left_data: IMU data from left sensor
            right_data: IMU data from right sensor
            
        Returns:
            Dictionary containing symmetry metrics
        """
        # Detect steps for both sides
        left_peaks, left_metrics = self.detect_gait_events(left_data['shank_acc_z'].values, left_data['foot_acc_z'].values)
        right_peaks, right_metrics = self.detect_gait_events(right_data['shank_acc_z'].values, right_data['foot_acc_z'].values)
        
        # Calculate symmetry indices
        symmetry = {}
        
        # Step time symmetry
        left_step_times = np.diff(left_peaks['heel_strikes']) / self.sampling_rate
        right_step_times = np.diff(right_peaks['heel_strikes']) / self.sampling_rate
        
        if len(left_step_times) > 0 and len(right_step_times) > 0:
            step_time_symmetry = (
                abs(np.mean(left_step_times) - np.mean(right_step_times)) /
                (0.5 * (np.mean(left_step_times) + np.mean(right_step_times)))
            ) * 100
        else:
            step_time_symmetry = 0
            
        symmetry['step_time_symmetry'] = step_time_symmetry
        
        # Calculate other symmetry metrics
        left_params = self.calculate_stride_parameters(left_data['shank_acc_z'].values, left_peaks['heel_strikes'])
        right_params = self.calculate_stride_parameters(right_data['shank_acc_z'].values, right_peaks['heel_strikes'])
        
        for param in ['stride_length', 'stride_time', 'stride_velocity']:
            if left_params[param] + right_params[param] > 0:
                symmetry[f'{param}_symmetry'] = (
                    abs(left_params[param] - right_params[param]) /
                    (0.5 * (left_params[param] + right_params[param]))
                ) * 100
            else:
                symmetry[f'{param}_symmetry'] = 0
                
        return symmetry

if __name__ == "__main__":
    # Load test data using absolute path
    data_path = "/Users/wangxiang/Desktop/my_workspace/mobisense/software-group/data-working/assets/feb16exp/2025-02-16_18-15-14_round_1.csv"
    data = pd.read_csv(data_path)
    
    # Rename columns for processing
    data = data.rename(columns={
        'imu0_acc_x': 'shank_acc_x',
        'imu0_acc_y': 'shank_acc_y',
        'imu0_acc_z': 'shank_acc_z',
        'imu1_acc_x': 'foot_acc_x',
        'imu1_acc_y': 'foot_acc_y',
        'imu1_acc_z': 'foot_acc_z'
    })
    
    # Initialize analyzer
    gait_analyzer = GaitAnalyzer()
    
    # Detect gait events
    events = gait_analyzer.detect_gait_events(
        data[['shank_acc_x', 'shank_acc_y', 'shank_acc_z']].values,
        data[['foot_acc_x', 'foot_acc_y', 'foot_acc_z']].values
    )
    
    # Create dummy angles data for testing
    angles = pd.DataFrame({
        'sagittal_angle': np.zeros(len(data)),
        'frontal_angle': np.zeros(len(data))
    })
    
    # Calculate gait parameters
    gait_params = gait_analyzer.calculate_gait_parameters(events, angles)
    pathological_metrics = gait_analyzer.analyze_pathological_gait(angles, events)
    
    # Print results
    print("\nGait Events Detected:")
    print("-------------------")
    print(f"Heel Strikes: {len(events['heel_strikes'])}")
    print(f"Toe Offs: {len(events['toe_offs'])}")
    
    print("\nGait Parameters:")
    print("---------------")
    for key, value in gait_params.items():
        print(f"{key}: {value:.2f}")
    
    print("\nPathological Metrics:")
    print("-------------------")
    for key, value in pathological_metrics.items():
        print(f"{key}: {value:.2f}")
    
    # Plot results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(data['shank_acc_z'], label='Shank Vertical Acc')
    plt.plot(data['foot_acc_z'], label='Foot Vertical Acc')
    plt.scatter(events['heel_strikes'], 
               data['shank_acc_z'].iloc[events['heel_strikes']], 
               color='red', label='Heel Strikes')
    plt.scatter(events['toe_offs'], 
               data['foot_acc_z'].iloc[events['toe_offs']], 
               color='green', label='Toe Offs')
    plt.title('Gait Event Detection')
    plt.xlabel('Samples')
    plt.ylabel('Acceleration (m/s²)')
    plt.legend()
    plt.grid(True)
    plt.show()
