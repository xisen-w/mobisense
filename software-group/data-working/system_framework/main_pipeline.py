import pandas as pd
from experiments import Participant, IMU_Experiment_Setup
from preprocessing import Preprocessor
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Main pipeline for the system

IMU_Data_Path = "/Users/wangxiang/Desktop/my_workspace/mobisense/software-group/data-working/assets/feb10exp/feb-10-fran-walking.csv"

# 1. Get the participant data
participant = Participant(participant_id="Fran", height=1.70, weight=70, age=20, gender="male", stride_length=0.5, stride_number_per_minute=96)

#  2. Get the experiment setup
experiment_setup = IMU_Experiment_Setup(experiment_name="Walking-3", experiment_description="Fran's own IMU-fixing pattern", participant=participant, experiment_data_path=IMU_Data_Path)

# 3. Load and transform the data
raw_data = pd.read_csv(IMU_Data_Path)

# Define numeric columns
numeric_columns = [
    'imu0_acc_x', 'imu0_acc_y', 'imu0_acc_z',
    'imu0_gyro_x', 'imu0_gyro_y', 'imu0_gyro_z',
    'imu1_acc_x', 'imu1_acc_y', 'imu1_acc_z',
    'imu1_gyro_x', 'imu1_gyro_y', 'imu1_gyro_z'
]

# Convert numeric columns to float
for col in numeric_columns:
    raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce')

# Split data for each IMU
imu0_numeric_data = pd.DataFrame({
    'acc_x': raw_data['imu0_acc_x'].astype(float),
    'acc_y': raw_data['imu0_acc_y'].astype(float),
    'acc_z': raw_data['imu0_acc_z'].astype(float),
    'gyro_x': raw_data['imu0_gyro_x'].astype(float),
    'gyro_y': raw_data['imu0_gyro_y'].astype(float),
    'gyro_z': raw_data['imu0_gyro_z'].astype(float),
})

imu1_numeric_data = pd.DataFrame({
    'acc_x': raw_data['imu1_acc_x'].astype(float),
    'acc_y': raw_data['imu1_acc_y'].astype(float),
    'acc_z': raw_data['imu1_acc_z'].astype(float),
    'gyro_x': raw_data['imu1_gyro_x'].astype(float),
    'gyro_y': raw_data['imu1_gyro_y'].astype(float),
    'gyro_z': raw_data['imu1_gyro_z'].astype(float),
})

# Store timestamps separately
imu0_timestamp = raw_data['imu0_timestamp']
imu1_timestamp = raw_data['imu1_timestamp']

print("Data loaded and converted to pandas dataframe.")
print("IMU0 data types:", imu0_numeric_data.dtypes)
print("IMU1 data types:", imu1_numeric_data.dtypes)

# Create preprocessors for each IMU
preprocessor_imu0 = Preprocessor(experiment_setup)
preprocessor_imu1 = Preprocessor(experiment_setup)

# 4. Process IMU0 data
eda_results_imu0 = preprocessor_imu0.perform_eda(imu0_numeric_data)
orientation_imu0 = preprocessor_imu0.get_orientation(imu0_numeric_data)

# Process IMU1 data
eda_results_imu1 = preprocessor_imu1.perform_eda(imu1_numeric_data)
print(eda_results_imu1)
orientation_imu1 = preprocessor_imu1.get_orientation(imu1_numeric_data)
print(orientation_imu1)

# 4.5 Filtering the data (for IMU0)
filtered_data_imu0 = preprocessor_imu0.low_pass_filter(imu0_numeric_data, cutoff_frequency=10)
filtered_data2_imu0 = preprocessor_imu0.gradient_filter(filtered_data_imu0, cutoff_frequency=0.5)
filtered_data3_imu0 = preprocessor_imu0.wavelet_filter(filtered_data2_imu0, wavelet='db4', level=2)
filtered_data4_imu0 = preprocessor_imu0.fft_filter(filtered_data3_imu0, cutoff_frequency=0.5)
filtered_data5_imu0 = preprocessor_imu0.kalman_filter(filtered_data4_imu0)

# Filtering for IMU1
filtered_data_imu1 = preprocessor_imu1.low_pass_filter(imu1_numeric_data, cutoff_frequency=10)
filtered_data2_imu1 = preprocessor_imu1.gradient_filter(filtered_data_imu1, cutoff_frequency=0.5)
filtered_data3_imu1 = preprocessor_imu1.wavelet_filter(filtered_data2_imu1, wavelet='db4', level=2)
filtered_data4_imu1 = preprocessor_imu1.fft_filter(filtered_data3_imu1, cutoff_frequency=0.5)
filtered_data5_imu1 = preprocessor_imu1.kalman_filter(filtered_data4_imu1)

# After filtering, you can add timestamps back if needed
filtered_data5_imu0['timestamp'] = imu0_timestamp
filtered_data5_imu1['timestamp'] = imu1_timestamp

# 5. Analyze the data
# TODO: Add analysis for both IMUs

# 6. Save the data
# TODO: Add saving logic for both IMUs

# 7. Visualize the data
# Option 2: Set seaborn defaults without using matplotlib style
sns.set_theme(style="whitegrid", palette="muted")

def plot_filtered_data_comparison(original, filtered_versions, sensor_type, imu_number):
    """Plot comparison of different filtering methods for each axis"""
    axes = ['x', 'y', 'z']
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 1, figure=fig)
    
    for idx, axis in enumerate(axes):
        ax = fig.add_subplot(gs[idx])
        col_name = f'{sensor_type}_{axis}'
        
        # Plot original data
        ax.plot(original[col_name], label='Original', alpha=0.5)
        
        # Plot filtered versions
        ax.plot(filtered_versions[0][col_name], label='Low Pass', alpha=0.7)
        ax.plot(filtered_versions[1][col_name], label='Gradient', alpha=0.7)
        ax.plot(filtered_versions[2][col_name], label='Wavelet', alpha=0.7)
        ax.plot(filtered_versions[3][col_name], label='FFT', alpha=0.7)
        ax.plot(filtered_versions[4][col_name], label='Kalman', alpha=0.7)
        
        ax.set_title(f'{sensor_type.upper()} {axis}-axis (IMU {imu_number})')
        ax.set_ylabel('Value')
        ax.legend()
    
    plt.tight_layout()
    return fig

def plot_orientation_data(orientation_data, imu_number):
    """Plot orientation angles over time"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(orientation_data['roll'], label='Roll')
    ax.plot(orientation_data['pitch'], label='Pitch')
    ax.plot(orientation_data['yaw'], label='Yaw')
    
    ax.set_title(f'Orientation Angles Over Time (IMU {imu_number})')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Degrees')
    ax.legend()
    
    return fig

def plot_eda_results(eda_results, imu_number):
    """Plot key EDA insights"""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Basic statistics plot
    ax1 = fig.add_subplot(gs[0, 0])
    stats_df = pd.DataFrame(eda_results['basic_stats'])
    sns.boxplot(data=stats_df, ax=ax1)
    ax1.set_title('Basic Statistics Distribution')
    ax1.tick_params(axis='x', rotation=45)
    
    # Frequency domain plot
    ax2 = fig.add_subplot(gs[0, 1])
    for sensor in ['acc_x', 'acc_y', 'acc_z']:
        freqs = eda_results['frequency_domain'][sensor]['dominant_frequencies']
        mags = eda_results['frequency_domain'][sensor]['frequency_magnitudes']
        ax2.plot(freqs, mags, label=sensor)
    ax2.set_title('Dominant Frequencies')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.legend()
    
    # Distribution metrics
    ax3 = fig.add_subplot(gs[1, 0])
    dist_metrics = pd.DataFrame({
        'Skewness': eda_results['distribution']['skewness'],
        'Kurtosis': eda_results['distribution']['kurtosis']
    })
    dist_metrics.plot(kind='bar', ax=ax3)
    ax3.set_title('Distribution Metrics')
    ax3.tick_params(axis='x', rotation=45)
    
    # Peak analysis
    ax4 = fig.add_subplot(gs[1, 1])
    peak_counts = {k: v['peak_count'] for k, v in eda_results['peaks'].items()}
    pd.Series(peak_counts).plot(kind='bar', ax=ax4)
    ax4.set_title('Peak Counts')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

# Create and save visualizations
# Filtered data comparison for accelerometer and gyroscope
filtered_versions_imu0 = [filtered_data_imu0, filtered_data2_imu0, filtered_data3_imu0, 
                         filtered_data4_imu0, filtered_data5_imu0]
filtered_versions_imu1 = [filtered_data_imu1, filtered_data2_imu1, filtered_data3_imu1, 
                         filtered_data4_imu1, filtered_data5_imu1]

# Plot and save IMU0 visualizations
fig_acc_imu0 = plot_filtered_data_comparison(imu0_numeric_data, filtered_versions_imu0, 'acc', 0)
fig_acc_imu0.savefig('imu0_accelerometer_comparison.png')

fig_gyro_imu0 = plot_filtered_data_comparison(imu0_numeric_data, filtered_versions_imu0, 'gyro', 0)
fig_gyro_imu0.savefig('imu0_gyroscope_comparison.png')

fig_orientation_imu0 = plot_orientation_data(orientation_imu0, 0)
fig_orientation_imu0.savefig('imu0_orientation.png')

fig_eda_imu0 = plot_eda_results(eda_results_imu0, 0)
fig_eda_imu0.savefig('imu0_eda_analysis.png')

# Plot and save IMU1 visualizations
fig_acc_imu1 = plot_filtered_data_comparison(imu1_numeric_data, filtered_versions_imu1, 'acc', 1)
fig_acc_imu1.savefig('imu1_accelerometer_comparison.png')

fig_gyro_imu1 = plot_filtered_data_comparison(imu1_numeric_data, filtered_versions_imu1, 'gyro', 1)
fig_gyro_imu1.savefig('imu1_gyroscope_comparison.png')

fig_orientation_imu1 = plot_orientation_data(orientation_imu1, 1)
fig_orientation_imu1.savefig('imu1_orientation.png')

fig_eda_imu1 = plot_eda_results(eda_results_imu1, 1)
fig_eda_imu1.savefig('imu1_eda_analysis.png')

plt.show()

