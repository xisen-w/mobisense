import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import iqr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from dtaidistance import dtw
from scipy.signal import find_peaks

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
            
        features = np.array(features)
        
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
        Returns dorsiflexion and plantarflexion angles
        """
        # Convert angular velocity to angle through integration
        angle = np.cumsum(gyro_data) / self.sampling_rate
        
        # Apply high-pass filter to remove drift
        b, a = signal.butter(2, 0.5/(self.sampling_rate/2), 'high')
        angle = signal.filtfilt(b, a, angle)
        
        rom_results = []
        
        # Analyze ROM in walking windows
        for i in range(len(is_walking)):
            if is_walking[i]:
                start_idx = i * (self.window_size - self.overlap)
                end_idx = start_idx + self.window_size
                window_angle = angle[start_idx:end_idx]
                
                # Find peaks (plantarflexion) and troughs (dorsiflexion)
                peaks, _ = find_peaks(window_angle)
                troughs, _ = find_peaks(-window_angle)
                
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
        
        return pd.DataFrame(rom_results)

    def plot_analysis(self, time, gyro_data, acc_data, is_walking, rom_results):
        """Visualize the analysis results"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Raw data and walking detection
        ax1.plot(time, gyro_data, label='Angular Velocity')
        walking_periods = np.where(is_walking)[0]
        for period in walking_periods:
            start_idx = period * (self.window_size - self.overlap)
            end_idx = start_idx + self.window_size
            ax1.axvspan(time[start_idx], time[end_idx], alpha=0.3, color='green')
        ax1.set_title('Gait Detection')
        ax1.set_ylabel('Angular Velocity (rad/s)')
        ax1.legend()
        
        # Plot 2: ROM analysis
        ax2.plot(time, np.cumsum(gyro_data)/self.sampling_rate, label='Ankle Angle')
        if len(rom_results) > 0:
            ax2.scatter(time[rom_results['window_start']], 
                       rom_results['plantarflexion'], 
                       color='red', label='Plantarflexion')
            ax2.scatter(time[rom_results['window_start']], 
                       rom_results['dorsiflexion'], 
                       color='blue', label='Dorsiflexion')
        ax2.set_title('Range of Motion Analysis')
        ax2.set_ylabel('Angle (rad)')
        ax2.legend()
        
        # Plot 3: ROM statistics
        if len(rom_results) > 0:
            ax3.boxplot([rom_results['plantarflexion'], 
                        rom_results['dorsiflexion'], 
                        rom_results['rom']], 
                       labels=['Plantarflexion', 'Dorsiflexion', 'ROM'])
            ax3.set_title('ROM Statistics')
            ax3.set_ylabel('Angle (rad)')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Load your data
    lateral_data = pd.read_csv('synthetic_gait_data_lateral_ankle.csv')
    medial_data = pd.read_csv('synthetic_gait_data_medial_ankle.csv')
    
    # Initialize analyzer
    analyzer = GaitAnalyzer(sampling_rate=100)
    
    # Analyze lateral ankle
    is_walking_lateral = analyzer.detect_walking(
        lateral_data['gyro_y'],  # Assuming Y-axis corresponds to dorsi/plantarflexion
        lateral_data['acc_z']
    )
    
    rom_results_lateral = analyzer.analyze_rom(
        lateral_data['gyro_y'],
        is_walking_lateral
    )
    
    # Plot results
    analyzer.plot_analysis(
        lateral_data['time'],
        lateral_data['gyro_y'],
        lateral_data['acc_z'],
        is_walking_lateral,
        rom_results_lateral
    )
    
    # Print summary statistics
    if len(rom_results_lateral) > 0:
        print("\nRange of Motion Statistics (Lateral Ankle):")
        print(f"Average Plantarflexion: {rom_results_lateral['plantarflexion'].mean():.2f} rad")
        print(f"Average Dorsiflexion: {rom_results_lateral['dorsiflexion'].mean():.2f} rad")
        print(f"Average ROM: {rom_results_lateral['rom'].mean():.2f} rad")