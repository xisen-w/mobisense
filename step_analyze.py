import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gaitmap.event_detection import RamppEventDetection

# Load the synthetic gait data
data = pd.read_csv('synthetic_gait_data.csv')

# Assuming the accelerometer data is in g and gyroscope data in degrees per second
# Convert gyroscope data to radians per second for gaitmap
data['gyro_x'] = np.deg2rad(data['gyro_x'])
data['gyro_y'] = np.deg2rad(data['gyro_y'])
data['gyro_z'] = np.deg2rad(data['gyro_z'])

# Prepare data in the expected format with correct anatomical direction names
imu_data = {
    'sensor': pd.DataFrame({
        'acc_pa': data['acc_x'],  # front-back
        'acc_ml': data['acc_y'],  # left-right
        'acc_si': data['acc_z'],  # up-down
        'gyr_pa': data['gyro_x'], # rotation around front-back axis
        'gyr_ml': data['gyro_y'], # rotation around left-right axis
        'gyr_si': data['gyro_z']  # rotation around up-down axis
    })
}

# Initialize the event detection algorithm
ed = RamppEventDetection()

# Create a proper stride list with stride IDs
stride_list = {
    'sensor': pd.DataFrame({
        's_id': [0],  # stride identifier
        'start': [0],
        'end': [len(data) - 1]
    })
}

# Detect gait events
ed = ed.detect(data=imu_data, sampling_rate_hz=100, stride_list=stride_list)

# Retrieve detected events
events = ed.min_vel_event_list_['sensor']
print("Detected events:", events)  # Add this to debug

# Create multiple subplots to visualize different signals
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Plot accelerometer data
ax1.plot(data['time'], imu_data['sensor']['acc_si'], label='Vertical Acceleration')
if not events['ic'].empty:
    ax1.scatter(data['time'][events['ic'].astype(int)], 
                imu_data['sensor']['acc_si'][events['ic'].astype(int)], 
                color='red', label='Initial Contact')
if not events['tc'].empty:
    ax1.scatter(data['time'][events['tc'].astype(int)], 
                imu_data['sensor']['acc_si'][events['tc'].astype(int)], 
                color='green', label='Terminal Contact')
ax1.set_title('Vertical Acceleration with Gait Events')
ax1.set_ylabel('Acceleration (g)')
ax1.legend()

# Plot gyroscope data
ax2.plot(data['time'], imu_data['sensor']['gyr_ml'], label='ML Angular Velocity')
if not events['ic'].empty:
    ax2.scatter(data['time'][events['ic'].astype(int)], 
                imu_data['sensor']['gyr_ml'][events['ic'].astype(int)], 
                color='red', label='Initial Contact')
if not events['tc'].empty:
    ax2.scatter(data['time'][events['tc'].astype(int)], 
                imu_data['sensor']['gyr_ml'][events['tc'].astype(int)], 
                color='green', label='Terminal Contact')
ax2.set_title('Medial-Lateral Angular Velocity with Gait Events')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angular Velocity (rad/s)')
ax2.legend()

plt.tight_layout()
plt.show()