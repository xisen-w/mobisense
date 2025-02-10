import pandas as pd
from experiments import Participant, IMU_Experiment_Setup
from preprocessing import Preprocessor

# Main pipeline for the system

IMU_Data_Path = "/Users/wangxiang/Desktop/my_workspace/mobisense/software-group/data-working/assets/jan23exp/IMU/xisen_walking_2025-01-22_17-46-40.csv"

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
imu0_data = pd.DataFrame({
    'acc_x': raw_data['imu0_acc_x'].astype(float),
    'acc_y': raw_data['imu0_acc_y'].astype(float),
    'acc_z': raw_data['imu0_acc_z'].astype(float),
    'gyro_x': raw_data['imu0_gyro_x'].astype(float),
    'gyro_y': raw_data['imu0_gyro_y'].astype(float),
    'gyro_z': raw_data['imu0_gyro_z'].astype(float),
    'timestamp': raw_data['imu0_timestamp']
})

imu1_data = pd.DataFrame({
    'acc_x': raw_data['imu1_acc_x'].astype(float),
    'acc_y': raw_data['imu1_acc_y'].astype(float),
    'acc_z': raw_data['imu1_acc_z'].astype(float),
    'gyro_x': raw_data['imu1_gyro_x'].astype(float),
    'gyro_y': raw_data['imu1_gyro_y'].astype(float),
    'gyro_z': raw_data['imu1_gyro_z'].astype(float),
    'timestamp': raw_data['imu1_timestamp']
})

print("Data loaded and converted to pandas dataframe.")
print("IMU0 data types:", imu0_data.dtypes)
print("IMU1 data types:", imu1_data.dtypes)

# Create preprocessors for each IMU
preprocessor_imu0 = Preprocessor(experiment_setup)
preprocessor_imu1 = Preprocessor(experiment_setup)

# 4. Process IMU0 data
eda_results_imu0 = preprocessor_imu0.perform_eda(imu0_data)
orientation_imu0 = preprocessor_imu0.get_orientation(imu0_data)

# Process IMU1 data
eda_results_imu1 = preprocessor_imu1.perform_eda(imu1_data)
print(eda_results_imu1)
orientation_imu1 = preprocessor_imu1.get_orientation(imu1_data)
print(orientation_imu1)

# 4.5 Filtering the data (for IMU0)
filtered_data_imu0 = preprocessor_imu0.low_pass_filter(imu0_data, cutoff_frequency=10)
filtered_data2_imu0 = preprocessor_imu0.gradient_filter(filtered_data_imu0, cutoff_frequency=0.5)
filtered_data3_imu0 = preprocessor_imu0.wavelet_filter(filtered_data2_imu0, wavelet='db4', level=3)
filtered_data4_imu0 = preprocessor_imu0.fft_filter(filtered_data3_imu0, cutoff_frequency=0.5)
filtered_data5_imu0 = preprocessor_imu0.kalman_filter(filtered_data4_imu0, cutoff_frequency=0.5)

# Filtering for IMU1
filtered_data_imu1 = preprocessor_imu1.low_pass_filter(imu1_data, cutoff_frequency=10)
filtered_data2_imu1 = preprocessor_imu1.gradient_filter(filtered_data_imu1, cutoff_frequency=0.5)
filtered_data3_imu1 = preprocessor_imu1.wavelet_filter(filtered_data2_imu1, wavelet='db4', level=3)
filtered_data4_imu1 = preprocessor_imu1.fft_filter(filtered_data3_imu1, cutoff_frequency=0.5)
filtered_data5_imu1 = preprocessor_imu1.kalman_filter(filtered_data4_imu1, cutoff_frequency=0.5)

# 5. Analyze the data
# TODO: Add analysis for both IMUs

# 6. Save the data
# TODO: Add saving logic for both IMUs

# 7. Visualize the data
# TODO: Add visualization for both IMUs

