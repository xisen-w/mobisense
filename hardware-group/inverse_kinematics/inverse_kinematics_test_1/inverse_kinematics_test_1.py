# Sample script to calculate angles in the sagittal plane.
# Note: This script works only under the following conditions:  
# 1. The IMUs' reference axes are aligned such that:  
#    - The x-axis points away from the treadmill.  
#    - The z-axis points towards the ceiling.  
# 2. The IMUs are positioned on the limb so that:  
#    - The x-axis extends away from the treadmill.  
#    - The z-axis points away from the body.
# Literature values for walking gait are from https://pubmed.ncbi.nlm.nih.gov/24998405/ 
# Literature values for weightbearing dorsiflexion ROM are from https://doi.org/10.7547/87507315-83-5-251 

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def load_selected_rows(csv_path, selected_rows):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Select relevant columns
    columns = ['imu0_roll', 'imu1_roll', 'imu0_timestamp']
    
    # Extract and print selected rows
    selected_data = df.loc[selected_rows, columns]
    
    return selected_data

# The plantarflexion formula compensates for drift, as the IMU takes around 45 seconds to stabilize.
# This delay occurs because the DMP (Digital Motion Processor) uses a fusion algorithm (e.g., Kalman filter),
# which requires time to converge and provide accurate data.
def calculate_plantar_flexion_angle(roll0, roll1, roll0_drift, roll1_drift):
    return (roll1 - roll1_drift) - (roll0 - roll0_drift) + 180

# Convert timestamps to seconds from the first timestamp
def convert_timestamps_to_seconds(timestamps):
    # Ensure the input is a list or a Pandas Series, then convert to a list if necessary
    if isinstance(timestamps, pd.Series):
        timestamps = timestamps.tolist()
    
    start_time = datetime.strptime(timestamps[0], "%Y-%m-%dT%H:%M:%S.%f")
    
    time_differences = [
        (datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f") - start_time).total_seconds()
        for ts in timestamps
    ]
    return time_differences

# Example usage:
csv_file = "HTWeek3-2025-02-03_15-23-39.csv"  # Replace with your actual file path

# Drift angle takes 45 seconds to stabilise. 
# This data was recorded at line 185.
data_drift= load_selected_rows(csv_file, 185)
roll0_drift = data_drift['imu0_roll']
roll1_drift = data_drift['imu1_roll']

# The angle between the foot and tibia when standing represents the rest angles, i.e., the angles when a person is standing.
# This data was recorded at line 595.
data_rest = load_selected_rows(csv_file, 595)
roll0_rest = data_rest['imu0_roll']
roll1_rest = data_rest['imu1_roll']
sagittal_angle_rest = calculate_plantar_flexion_angle(roll0_rest, roll1_rest, roll0_drift , roll1_drift)

# The angle between the foot and tibia during a deep squat.
# This data was recorded at line 621.
data_squat = load_selected_rows(csv_file, 621)
roll0_squat = data_squat['imu0_roll']
roll1_squat = data_squat['imu1_roll']
sagittal_angle_squat = calculate_plantar_flexion_angle(roll0_squat,roll1_squat, roll0_drift , roll1_drift)

# The angle between the foot and tibia while pointing foot.
# This data was recorded at line 644.
data_point = load_selected_rows(csv_file, 644)
roll0_point = data_point['imu0_roll']
roll1_point = data_point['imu1_roll']
sagittal_angle_point = calculate_plantar_flexion_angle(roll0_point,roll1_point, roll0_drift , roll1_drift)

# The angle between the foot and tibia while taking two steps.
# This data was recorded between lines 680 to 690.
data_walk = load_selected_rows(csv_file, list(range(680, 691)))
roll0_walk = data_walk['imu0_roll']
roll1_walk = data_walk['imu1_roll']
timestamp_walk = data_walk['imu0_timestamp']
time = convert_timestamps_to_seconds(timestamp_walk)
sagittal_angle_walk = calculate_plantar_flexion_angle(roll0_walk,roll1_walk, roll0_drift , roll1_drift)

# Calculate plantar flexion angle during walking by subtracting the resting angle
plantar_flexion_angle_walk = sagittal_angle_rest - sagittal_angle_walk

# Plotting time vs plantar flexion angles
plt.plot(time, plantar_flexion_angle_walk)
plt.xlabel('Time (s)')
plt.ylabel('Plantar Flexion Angle (degrees)')
plt.title('Ankle sagittal plane angles (dorsiflexion [+] and plantar flexion [−])')
plt.legend()
plt.grid(True)
plt.show()
# Dorsiflexion: 13.6° (± 3.5°) literature vs. 15.7° in code (reasonable match)  
# Plantar flexion: -18.7° (± 8.0°) literature vs. -7.78° in code (sampling error)  
# Note: Plantar flexion during the swing phase is shorter, affecting the recorded value.

# Print the results for verification
print(sagittal_angle_rest - sagittal_angle_squat)
# Literature value is 7.1 to 34.7 degrees for weightbearing test dorsiflexion

print(sagittal_angle_rest - sagittal_angle_point)
# Literature value is 20 to 50 degrees according to Medical News Today 








