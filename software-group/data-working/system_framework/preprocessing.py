import numpy as np
import pandas as pd
from scipy import signal
import pywt
from experiments import Participant, IMU_Experiment_Setup
from typing import Optional, Union, List

class Preprocessor:
    def __init__(self, experiment_setup: Optional[IMU_Experiment_Setup] = None):
        self.experiment_setup = experiment_setup

    def get_available_data(self) -> pd.DataFrame:
        """
        Retrieves data from the experiment setup.
        Returns:
            pd.DataFrame: Combined participant and experiment data
        """
        if self.experiment_setup is None:
            raise ValueError("Experiment setup not initialized")
            
        participant_data = self.experiment_setup.participant.get_participant_data()
        data_path = self.experiment_setup.experiment_data_path
        # TODO: Implement data loading from data_path
        return participant_data

    # Data preprocessing utilities
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes data using z-score normalization
        """
        return (data - data.mean()) / data.std()

    def remove_outliers(self, data: pd.DataFrame, threshold: float = 3) -> pd.DataFrame:
        """
        Removes outliers using z-score method
        """
        z_scores = np.abs((data - data.mean()) / data.std())
        return data[z_scores < threshold]

    # Filtering methods
    def roll_mean_filter(self, data: pd.DataFrame, window_size: int) -> pd.DataFrame:
        """
        Applies rolling mean filter to smooth the data
        """
        return data.rolling(window=window_size, center=True).mean()

    def low_pass_filter(self, data: pd.DataFrame, cutoff_frequency: float, 
                       sampling_rate: float = 100.0) -> pd.DataFrame:
        """
        Applies Butterworth low-pass filter
        """
        nyquist = sampling_rate * 0.5
        normal_cutoff = cutoff_frequency / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
        
        filtered_data = pd.DataFrame(index=data.index)
        for column in data.columns:
            filtered_data[column] = signal.filtfilt(b, a, data[column])
        return filtered_data

    def kalman_filter(self, data: pd.DataFrame, process_variance: float = 1e-5,
                     measurement_variance: float = 1e-2) -> pd.DataFrame:
        """
        Applies Kalman filter for state estimation
        """
        filtered_data = pd.DataFrame(index=data.index)
        
        for column in data.columns:
            measurements = data[column].values
            n_iterations = len(measurements)
            
            # Initial state
            post_estimate = measurements[0]
            post_error_estimate = 1.0
            
            estimates = np.zeros(n_iterations)
            
            for i in range(n_iterations):
                # Prediction
                prior_estimate = post_estimate
                prior_error_estimate = post_error_estimate + process_variance
                
                # Update
                kalman_gain = prior_error_estimate / (prior_error_estimate + measurement_variance)
                post_estimate = prior_estimate + kalman_gain * (measurements[i] - prior_estimate)
                post_error_estimate = (1 - kalman_gain) * prior_error_estimate
                
                estimates[i] = post_estimate
                
            filtered_data[column] = estimates
            
        return filtered_data

    def wavelet_filter(self, data: pd.DataFrame, wavelet: str = 'db4',
                      level: int = 3) -> pd.DataFrame:
        """
        Applies wavelet filter for multi-resolution analysis
        """
        filtered_data = pd.DataFrame(index=data.index)
        
        for column in data.columns:
            # Get original length
            original_length = len(data[column])
            
            # Perform wavelet decomposition and reconstruction
            coeffs = pywt.wavedec(data[column].values, wavelet, level=level)
            reconstructed = pywt.waverec(coeffs, wavelet)
            
            # Ensure the reconstructed signal matches the original length
            if len(reconstructed) > original_length:
                reconstructed = reconstructed[:original_length]
            elif len(reconstructed) < original_length:
                # Pad with the last value if necessary
                reconstructed = np.pad(reconstructed, 
                                     (0, original_length - len(reconstructed)),
                                     'edge')
            
            filtered_data[column] = reconstructed
            
        return filtered_data

    def wavelet_denoise(self, data: pd.DataFrame, wavelet: str = 'db4',
                       threshold_mode: str = 'soft') -> pd.DataFrame:
        """
        Applies wavelet denoising using threshold
        """
        filtered_data = pd.DataFrame(index=data.index)
        
        for column in data.columns:
            # Decompose signal
            coeffs = pywt.wavedec(data[column].values, wavelet)
            
            # Calculate threshold
            threshold = np.sqrt(2 * np.log(len(data[column])))
            
            # Apply threshold to coefficients
            new_coeffs = list(coeffs)
            for i in range(1, len(coeffs)):
                new_coeffs[i] = pywt.threshold(coeffs[i], threshold, threshold_mode)
            
            # Reconstruct signal
            filtered_data[column] = pywt.waverec(new_coeffs, wavelet)
            
        return filtered_data
    
    # Below are the functions for preprocessing the data using frequency

    def fft_filter(self, data: pd.DataFrame, cutoff_frequency: float, 
                   sampling_rate: float = 100.0) -> pd.DataFrame:
        """
        Applies Fourier Transform based filtering
        
        Args:
            data: Input DataFrame
            cutoff_frequency: Frequency above which to filter out (Hz)
            sampling_rate: Data sampling rate in Hz
            
        Returns:
            pd.DataFrame: Filtered data
        """
        filtered_data = pd.DataFrame(index=data.index)
        
        for column in data.columns:
            # Perform FFT
            signal_fft = np.fft.fft(data[column].values)
            frequencies = np.fft.fftfreq(len(data[column]), d=1/sampling_rate)
            
            # Create frequency mask
            mask = np.abs(frequencies) <= cutoff_frequency
            
            # Apply filter in frequency domain
            signal_fft_filtered = signal_fft * mask
            
            # Inverse FFT to get back to time domain
            filtered_signal = np.real(np.fft.ifft(signal_fft_filtered))
            
            filtered_data[column] = filtered_signal
            
        return filtered_data

    def gradient_filter(self, data: pd.DataFrame, cutoff_frequency: float,
                       sampling_rate: float = 100.0) -> pd.DataFrame:
        """
        Applies gradient-based frequency filtering
        
        Args:
            data: Input DataFrame
            cutoff_frequency: Frequency threshold for gradient filtering (Hz)
            sampling_rate: Data sampling rate in Hz
            
        Returns:
            pd.DataFrame: Filtered data
        """
        filtered_data = pd.DataFrame(index=data.index)
        
        # Time step
        dt = 1/sampling_rate
        
        for column in data.columns:
            values = data[column].values
            
            # Calculate gradient
            gradient = np.gradient(values, dt)
            
            # Apply FFT to gradient
            gradient_fft = np.fft.fft(gradient)
            frequencies = np.fft.fftfreq(len(gradient), dt)
            
            # Create frequency mask
            mask = np.abs(frequencies) <= cutoff_frequency
            
            # Filter gradient in frequency domain
            gradient_fft_filtered = gradient_fft * mask
            
            # Inverse FFT to get filtered gradient
            filtered_gradient = np.real(np.fft.ifft(gradient_fft_filtered))
            
            # Reconstruct signal by cumulative integration
            filtered_signal = np.cumsum(filtered_gradient) * dt
            
            # Remove DC offset to match original signal mean
            filtered_signal = filtered_signal - np.mean(filtered_signal) + np.mean(values)
            
            filtered_data[column] = filtered_signal
            
        return filtered_data

    def process_pipeline(self, data: pd.DataFrame, 
                        steps: List[dict]) -> pd.DataFrame:
        """
        Applies a sequence of preprocessing steps
        
        Args:
            data: Input DataFrame
            steps: List of dictionaries containing preprocessing steps and parameters
                  Example: [
                      {'method': 'normalize_data'},
                      {'method': 'low_pass_filter', 'params': {'cutoff_frequency': 10}},
                      {'method': 'kalman_filter'}
                  ]
        """
        processed_data = data.copy()
        
        for step in steps:
            method = step['method']
            params = step.get('params', {})
            
            if hasattr(self, method):
                processing_func = getattr(self, method)
                processed_data = processing_func(processed_data, **params)
            else:
                raise ValueError(f"Unknown preprocessing method: {method}")
                
        return processed_data
    
    def perform_eda(self, data: pd.DataFrame) -> dict:
        """
        Performs Exploratory Data Analysis on IMU data
        
        Args:
            data: Input DataFrame with IMU measurements
            
        Returns:
            dict: Dictionary containing EDA results
        """
        eda_results = {}
        
        # Separate numeric and non-numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        timestamp_cols = data.select_dtypes(include=['datetime64', 'object']).columns
        
        # Basic statistics for numeric columns only
        if len(numeric_cols) > 0:
            eda_results['basic_stats'] = {
                'mean': data[numeric_cols].mean(),
                'std': data[numeric_cols].std(),
                'min': data[numeric_cols].min(),
                'max': data[numeric_cols].max(),
                'range': data[numeric_cols].max() - data[numeric_cols].min(),
                'median': data[numeric_cols].median()
            }
        
        # Time series analysis for timestamp columns
        if len(timestamp_cols) > 0:
            try:
                # Convert timestamp strings to datetime if they're not already
                time_data = data[timestamp_cols].apply(pd.to_datetime)
                eda_results['time_analysis'] = {
                    'start_time': time_data.min(),
                    'end_time': time_data.max(),
                    'duration': (time_data.max() - time_data.min()),
                    'sampling_intervals': time_data.diff().describe()
                }
            except Exception as e:
                print(f"Warning: Could not process timestamp columns: {e}")
        
        # Peak analysis for numeric columns
        if len(numeric_cols) > 0:
            eda_results['peaks'] = {}
            for column in numeric_cols:
                peaks, _ = signal.find_peaks(data[column], height=data[column].mean())
                valleys, _ = signal.find_peaks(-data[column], height=-data[column].mean())
                
                eda_results['peaks'][column] = {
                    'peak_indices': peaks,
                    'peak_values': data[column].iloc[peaks],
                    'valley_indices': valleys,
                    'valley_values': data[column].iloc[valleys],
                    'peak_count': len(peaks),
                    'valley_count': len(valleys)
                }
            
            # Distribution analysis for numeric columns
            eda_results['distribution'] = {
                'skewness': data[numeric_cols].skew(),
                'kurtosis': data[numeric_cols].kurtosis(),
                'quartiles': data[numeric_cols].quantile([0.25, 0.5, 0.75]),
                'iqr': data[numeric_cols].quantile(0.75) - data[numeric_cols].quantile(0.25)
            }
            
            # Frequency domain analysis
            if isinstance(data.index, pd.DatetimeIndex):
                sampling_rate = 1 / (data.index[1] - data.index[0]).total_seconds()
            else:
                sampling_rate = 100  # Default assumption
                
            eda_results['frequency_domain'] = {}
            for column in numeric_cols:
                fft_vals = np.fft.fft(data[column].values)
                freqs = np.fft.fftfreq(len(data), d=1/sampling_rate)
                
                # Get dominant frequencies
                dominant_freq_idx = np.argsort(np.abs(fft_vals))[-5:]  # Top 5 frequencies
                eda_results['frequency_domain'][column] = {
                    'dominant_frequencies': freqs[dominant_freq_idx],
                    'frequency_magnitudes': np.abs(fft_vals)[dominant_freq_idx]
                }
        
        return eda_results

    def get_orientation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates orientation from IMU data (accelerometer and gyroscope)
        
        Args:
            data: DataFrame with columns for accelerometer (acc_x, acc_y, acc_z) 
                 and optionally gyroscope (gyro_x, gyro_y, gyro_z)
                 
        Returns:
            pd.DataFrame: DataFrame with euler angles (roll, pitch, yaw)
        """
        # Check required columns
        required_cols = ['acc_x', 'acc_y', 'acc_z']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain accelerometer columns: {required_cols}")
        
        # Initialize results DataFrame
        orientation = pd.DataFrame(index=data.index)
        
        # Calculate roll and pitch from accelerometer
        acc_x = data['acc_x'].values
        acc_y = data['acc_y'].values
        acc_z = data['acc_z'].values
        
        # Roll (rotation around X-axis)
        orientation['roll'] = np.arctan2(acc_y, np.sqrt(acc_x**2 + acc_z**2))
        
        # Pitch (rotation around Y-axis)
        orientation['pitch'] = np.arctan2(-acc_x, np.sqrt(acc_y**2 + acc_z**2))
        
        # Yaw calculation (if gyroscope data is available)
        if all(col in data.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
            # Simple integration of gyroscope data for yaw
            # Note: This is a basic implementation and may drift over time
            dt = 1/100  # Assuming 100Hz sampling rate
            gyro_z = data['gyro_z'].values
            orientation['yaw'] = np.cumsum(gyro_z * dt)
            
            # Optional: Apply complementary filter if magnetometer data is available
            if all(col in data.columns for col in ['mag_x', 'mag_y', 'mag_z']):
                mag_x = data['mag_x'].values
                mag_y = data['mag_y'].values
                
                # Calculate magnetic heading
                mag_heading = np.arctan2(mag_y, mag_x)
                
                # Complementary filter
                alpha = 0.95  # Filter coefficient
                orientation['yaw'] = alpha * orientation['yaw'] + (1 - alpha) * mag_heading
        else:
            orientation['yaw'] = np.nan
            
        # Convert radians to degrees
        orientation = orientation * (180 / np.pi)
        
        return orientation

    def sampling__boosting(self, data: pd.DataFrame, target_freq: float, 
                          method: str = 'linear', fill_gaps: bool = True) -> pd.DataFrame:
        """
        Boosts the sampling rate of the data to a target frequency using interpolation
         
        Args:
            data: Input DataFrame with datetime index
            target_freq: Target frequency in Hz
            method: Interpolation method ('linear', 'cubic', 'spline', 'polynomial')
            fill_gaps: Whether to fill NaN values that might occur during resampling
            
        Returns:
            pd.DataFrame: Resampled data at target frequency
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")
        
        # Calculate target time delta
        target_delta = pd.Timedelta(seconds=1/target_freq)
        
        # Create new time index at target frequency
        new_index = pd.date_range(start=data.index[0], 
                                 end=data.index[-1], 
                                 freq=target_delta)
        
        # Resample and interpolate
        resampled_data = pd.DataFrame(index=new_index)
        
        for column in data.columns:
            if method == 'linear':
                resampled_data[column] = data[column].reindex(new_index).interpolate(method='linear')
            elif method == 'cubic':
                resampled_data[column] = data[column].reindex(new_index).interpolate(method='cubic')
            elif method == 'spline':
                resampled_data[column] = data[column].reindex(new_index).interpolate(method='spline', order=3)
            elif method == 'polynomial':
                resampled_data[column] = data[column].reindex(new_index).interpolate(method='polynomial', order=2)
            else:
                raise ValueError(f"Unsupported interpolation method: {method}")
        
        # Optionally fill any remaining gaps
        if fill_gaps:
            resampled_data = resampled_data.fillna(method='ffill').fillna(method='bfill')
        
        return resampled_data
    




    







