import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

""" Plots sagittal angle of ankle joints from DMP values """

# Constants
lines = 474 # Number of lines of CSV file
sagittal_angle_rest = 125 # Sagittal angle at rest in degrees (measured for Francesco)
csv_file = "/Users/francescobalanzoni/Documents/Python/MEng/3YP/mobisense/software-group/data-working/assets/mar12exp/2025-03-12_10-22-44-r3-limping2.csv"

def load_selected_rows(csv_path, selected_rows):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Select relevant columns
    columns = ['imu0_roll', 'imu1_roll', 'imu0_timestamp']
    
    # Extract and print selected rows
    selected_data = df.loc[selected_rows, columns]
    
    return selected_data

def calculate_plantar_flexion_angle(roll0, roll1):
    return roll1 - roll0 + 180

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

# The angle between the foot and tibia when standing represents the rest angles, i.e., the angles when a person is standing
# This was measured to be 125 during testing

# The angle between the foot and tibia while taking steps
data_walk = load_selected_rows(csv_file, list(range(1,lines)))
roll0_walk = data_walk['imu0_roll']
roll1_walk = data_walk['imu1_roll']
timestamp_walk = data_walk['imu0_timestamp']
time = convert_timestamps_to_seconds(timestamp_walk)
sagittal_angle_walk = calculate_plantar_flexion_angle(roll0_walk,roll1_walk)

# Calculate plantar flexion angle during walking by subtracting the resting angle
plantar_flexion_angle_walk = sagittal_angle_rest - sagittal_angle_walk

# Plotting time vs plantar flexion angles
plt.plot(time, plantar_flexion_angle_walk)
plt.xlabel('Time (s)')
plt.ylabel('Plantar Flexion Angle (degrees)')
plt.title('Ankle sagittal plane angles (dorsiflexion [+] and plantar flexion [−])')
plt.grid(True)
plt.show()