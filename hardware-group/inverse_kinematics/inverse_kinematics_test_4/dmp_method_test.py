import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import detrend

""" Plots sagittal angle of ankle joints from DMP values with detrending """

# Constants
lines = 966  # Number of lines of CSV file
sagittal_angle_rest = 125  # Sagittal angle at rest in degrees (measured for Francesco)
csv_file = "/Users/francescobalanzoni/Documents/Python/MEng/3YP/mobisense/software-group/data-working/assets/mar12exp/2025-03-12_10-25-01-r4-walking3.csv"

# Define the period in which you want to set values to a constant
t_start = 10  # Start time in seconds
t_end = 35   # End time in seconds

def load_selected_rows(csv_path, selected_rows):
    df = pd.read_csv(csv_path)
    columns = ['imu0_roll', 'imu1_roll', 'imu0_timestamp']
    selected_data = df.loc[selected_rows, columns]
    return selected_data

def calculate_plantar_flexion_angle(roll0, roll1):
    return roll1 - roll0 + 180

def convert_timestamps_to_seconds(timestamps):
    if isinstance(timestamps, pd.Series):
        timestamps = timestamps.tolist()
    
    start_time = datetime.strptime(timestamps[0], "%Y-%m-%dT%H:%M:%S.%f")
    
    return [(datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f") - start_time).total_seconds() for ts in timestamps]

# Load data
data_walk = load_selected_rows(csv_file, list(range(1, lines)))
roll0_walk = data_walk['imu0_roll']
roll1_walk = data_walk['imu1_roll']
timestamp_walk = data_walk['imu0_timestamp']

# Compute time and sagittal angle
time = convert_timestamps_to_seconds(timestamp_walk)
sagittal_angle_walk = calculate_plantar_flexion_angle(roll0_walk, roll1_walk)

# Compute plantar flexion angle
plantar_flexion_angle_walk = sagittal_angle_rest - sagittal_angle_walk

# 🛠 **Fix NaN/Inf issues before detrending**
plantar_flexion_angle_walk = np.array(plantar_flexion_angle_walk)

# Remove NaN and Inf values
valid_indices = ~np.isnan(plantar_flexion_angle_walk) & ~np.isinf(plantar_flexion_angle_walk)
time_cleaned = np.array(time)[valid_indices]
plantar_flexion_cleaned = plantar_flexion_angle_walk[valid_indices]

# Apply detrending filter
plantar_flexion_angle_detrended = detrend(plantar_flexion_cleaned)

# 📌 **Replace values in the selected time range with the value at t_start**
time_array = np.array(time_cleaned)
replacement_value = plantar_flexion_angle_detrended[np.where(time_array >= t_start)[0][0]]  # Get first value at t_start

# Modify values in the specified time range
mask = (time_array >= t_start) & (time_array <= t_end)
plantar_flexion_angle_detrended[mask] = replacement_value  # Set values to the starting value

# Plotting
plt.plot(time_array, plantar_flexion_angle_detrended, label='Processed Angle')
plt.axvspan(t_start, t_end, color='red', alpha=0.3, label='Modified Period')  # Highlight modified range
plt.xlabel('Time (s)')
plt.ylabel('Plantar Flexion Angle (degrees)')
plt.title('Ankle Sagittal Plane Angles (Dorsiflexion [+] & Plantar Flexion [−])')
plt.grid(True)
plt.legend()
plt.show()


