import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import iqr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from dtaidistance import dtw
from scipy.signal import find_peaks
import os

class GaitAnalyzer:
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate
        self.window_size = int(2 * sampling_rate)  # 2-second window
        self.overlap = int(self.window_size * 0.5)  # 50% overlap
        
    def detect_walking(self, gyro_data, acc_data):
        """
        Detect walking periods using multiple features
        Based on: "Assessment of Walking Features from Lower Trunk Accelerometry" 
        (Sejdić et al., 2016)
        """
        # Calculate features in sliding windows
        features = []
        
        # Convert input data to numpy arrays if they're pandas series
        gyro_data = np.array(gyro_data)
        acc_data = np.array(acc_data)
        
        for i in range(0, len(gyro_data) - self.window_size, self.window_size - self.overlap):
            window_gyro = gyro_data[i:i + self.window_size]
            window_acc = acc_data[i:i + self.window_size]
            
            # Feature 1: Dominant frequency (should be 1.4-2.5 Hz for walking)
            freqs, psd = signal.welch(window_gyro, fs=self.sampling_rate)
            dom_freq = freqs[np.argmax(psd)]
            
            # Feature 2: Signal energy
            energy = np.sum(np.square(window_gyro)) / len(window_gyro)
            
            # Feature 3: Step regularity using autocorrelation
            autocorr = np.correlate(window_gyro, window_gyro, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            peaks, _ = find_peaks(autocorr, distance=self.sampling_rate//2)
            if len(peaks) > 1:
                step_regularity = autocorr[peaks[1]] / autocorr[peaks[0]]
            else:
                step_regularity = 0
                
            # Feature 4: Acceleration variance
            acc_var = np.var(window_acc)
            
            features.append([dom_freq, energy, step_regularity, acc_var])
        
        # Convert features list to numpy array and reshape if empty
        features = np.array(features)
        if len(features) == 0:
            return np.array([])  # Return empty array if no features were calculated
        
        # Ensure features is 2D array
        if features.ndim == 1:
            features = features.reshape(-1, 4)
        
        # Classify windows as walking/not walking using thresholds
        is_walking = np.all([
            (features[:, 0] >= 1.4) & (features[:, 0] <= 2.5),  # Normal walking frequency
            features[:, 1] > np.mean(features[:, 1]) * 0.5,     # Sufficient energy
            features[:, 2] > 0.5,                               # Regular steps
            features[:, 3] > np.mean(features[:, 3]) * 0.3      # Sufficient movement
        ], axis=0)
        
        return is_walking
    
    def analyze_rom(self, gyro_data, is_walking):
        """
        Analyze Range of Motion during walking periods
        Returns dorsiflexion and plantarflexion angles in degrees
        """
        # Convert input to numpy array if it's a pandas series
        gyro_data = np.array(gyro_data)
        
        # First, filter the gyro data to remove noise and drift
        # Using a bandpass filter between 0.3 Hz and 3 Hz (typical walking frequencies)
        nyquist = self.sampling_rate / 2
        low = 0.3 / nyquist
        high = 3.0 / nyquist
        b, a = signal.butter(2, [low, high], btype='band')
        filtered_gyro = signal.filtfilt(b, a, gyro_data)
        
        # Initialize angle array
        angle = np.zeros_like(filtered_gyro)
        
        # Integrate gyro data with reset at each stride to prevent drift
        stride_duration = int(self.sampling_rate)  # Assume 1-second stride
        
        for i in range(0, len(filtered_gyro), stride_duration):
            end_idx = min(i + stride_duration, len(filtered_gyro))
            # Trapezoidal integration using numpy's trapz
            angle[i:end_idx] = np.trapz(filtered_gyro[i:end_idx], 
                                      dx=1/self.sampling_rate)
        
        # Convert to degrees
        angle = np.rad2deg(angle)
        
        # Remove any remaining trend
        angle = signal.detrend(angle)
        
        rom_results = []
        
        # Analyze ROM in walking windows
        for i in range(len(is_walking)):
            if is_walking[i]:
                start_idx = i * (self.window_size - self.overlap)
                end_idx = start_idx + self.window_size
                window_angle = angle[start_idx:end_idx]
                
                # Find peaks and troughs with more specific parameters
                peaks, _ = find_peaks(window_angle, 
                                    distance=int(0.5 * self.sampling_rate),  # Min 0.5s between peaks
                                    prominence=2.0)  # Minimum 2 degrees prominence
                
                troughs, _ = find_peaks(-window_angle, 
                                      distance=int(0.5 * self.sampling_rate),
                                      prominence=2.0)
                
                if len(peaks) > 0 and len(troughs) > 0:
                    plantarflexion = np.mean(window_angle[peaks])
                    dorsiflexion = np.mean(window_angle[troughs])
                    rom = plantarflexion - dorsiflexion
                    
                    rom_results.append({
                        'window_start': start_idx,
                        'plantarflexion': plantarflexion,
                        'dorsiflexion': dorsiflexion,
                        'rom': rom
                    })
        
        # Add plotting for debugging
        plt.figure(figsize=(15, 10))
        plt.subplot(211)
        plt.plot(filtered_gyro, label='Filtered Gyro')
        plt.title('Filtered Gyro Signal')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(212)
        plt.plot(angle, label='Ankle Angle')
        plt.title('Calculated Ankle Angle (degrees)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return pd.DataFrame(rom_results)

    def plot_analysis(self, time, gyro_data, acc_data, is_walking, rom_results):
        """
        Plot the analysis results with improved visualization
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Convert time to seconds for better x-axis
        if isinstance(time[0], str):
            time = pd.to_datetime(time)
        time_sec = np.arange(len(time)) / self.sampling_rate
        
        # Plot 1: Angular velocity and walking detection
        ax1.plot(time_sec, gyro_data, label='Angular Velocity', color='blue')
        
        # Highlight walking periods
        for i in range(len(is_walking)):
            if is_walking[i]:
                start = i * (self.window_size - self.overlap) / self.sampling_rate
                end = (i * (self.window_size - self.overlap) + self.window_size) / self.sampling_rate
                ax1.axvspan(start, end, color='green', alpha=0.2)
        
        ax1.set_title('Gait Detection')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Angular Velocity (rad/s)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Range of Motion Analysis
        if not rom_results.empty:
            # Calculate and plot ankle angle
            angle = np.zeros_like(gyro_data)
            for i in range(1, len(gyro_data)):
                angle[i] = angle[i-1] + gyro_data[i] / self.sampling_rate
            angle = np.rad2deg(signal.detrend(angle))  # Convert to degrees and remove drift
            
            ax2.plot(time_sec, angle, label='Ankle Angle', color='blue')
            
            # Plot ROM points
            for _, row in rom_results.iterrows():
                t = row['window_start'] / self.sampling_rate
                ax2.plot(t, row['plantarflexion'], 'r^', label='Plantarflexion' if _ == 0 else '')
                ax2.plot(t, row['dorsiflexion'], 'gv', label='Dorsiflexion' if _ == 0 else '')
        
        ax2.set_title('Range of Motion Analysis')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angle (degrees)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Directory setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load your data using proper path joining
    fran_file = os.path.join(current_dir, 'assets/jan23exp/fran_walking_2025-01-22_17-21-12.csv')
    xisen_file = os.path.join(current_dir, 'assets/jan23exp/xisen_walking_2025-01-22_17-46-40.csv')
    
    fran_data = pd.read_csv(fran_file)
    xisen_data = pd.read_csv(xisen_file)
    
    # Initialize analyzer
    analyzer = GaitAnalyzer(sampling_rate=100)
    
    # Analyze Fran's data
    is_walking_fran = analyzer.detect_walking(
        fran_data['imu0_gyro_y'],  # Using gyro Y for dorsi/plantarflexion
        fran_data['imu0_acc_z']    # Using acc Z for vertical movement
    )
    
    rom_results_fran = analyzer.analyze_rom(
        fran_data['imu0_gyro_y'],
        is_walking_fran
    )
    
    # Plot Fran's results
    analyzer.plot_analysis(
        fran_data['imu0_timestamp'],
        fran_data['imu0_gyro_y'],
        fran_data['imu0_acc_z'],
        is_walking_fran,
        rom_results_fran
    )
    
    # Analyze Xisen's data
    is_walking_xisen = analyzer.detect_walking(
        xisen_data['imu0_gyro_y'],
        xisen_data['imu0_acc_z']
    )
    
    rom_results_xisen = analyzer.analyze_rom(
        xisen_data['imu0_gyro_y'],
        is_walking_xisen
    )
    
    # Plot Xisen's results
    analyzer.plot_analysis(
        xisen_data['imu0_timestamp'],
        xisen_data['imu0_gyro_y'],
        xisen_data['imu0_acc_z'],
        is_walking_xisen,
        rom_results_xisen
    )
    
    # Print summary statistics for both subjects
    print("\nRange of Motion Statistics (Fran):")
    if len(rom_results_fran) > 0:
        print(f"Average Plantarflexion: {rom_results_fran['plantarflexion'].mean():.2f} degrees")
        print(f"Average Dorsiflexion: {rom_results_fran['dorsiflexion'].mean():.2f} degrees")
        print(f"Average ROM: {rom_results_fran['rom'].mean():.2f} degrees")
    
    print("\nRange of Motion Statistics (Xisen):")
    if len(rom_results_xisen) > 0:
        print(f"Average Plantarflexion: {rom_results_xisen['plantarflexion'].mean():.2f} degrees")
        print(f"Average Dorsiflexion: {rom_results_xisen['dorsiflexion'].mean():.2f} degrees")
        print(f"Average ROM: {rom_results_xisen['rom'].mean():.2f} degrees")