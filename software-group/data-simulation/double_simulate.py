import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
sampling_rate = 100  # Hz
duration = 10  # seconds
time = np.linspace(0, duration, duration * sampling_rate)

def generate_imu_data(base_freq=1.5, phase_shift=0, amplitude_modifier=1.0):
    """Generate IMU data with slight variations for different sensor positions"""
    # Accelerometer data (g)
    acc_x = amplitude_modifier * 0.5 * np.sin(2 * np.pi * base_freq * time + phase_shift)
    acc_y = amplitude_modifier * 0.1 * np.sin(2 * np.pi * 3 * time + phase_shift)
    acc_z = 1.0 + amplitude_modifier * 0.2 * np.sin(2 * np.pi * base_freq * time + phase_shift)
    
    # Gyroscope data (degrees per second)
    gyro_x = amplitude_modifier * 5 * np.sin(2 * np.pi * base_freq * time + phase_shift)
    gyro_y = amplitude_modifier * 2 * np.sin(2 * np.pi * base_freq * time + phase_shift)
    gyro_z = amplitude_modifier * 3 * np.sin(2 * np.pi * base_freq * time + phase_shift)
    
    return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

# Generate data for two IMUs with slight differences
# IMU1 (lateral ankle)
acc_x1, acc_y1, acc_z1, gyro_x1, gyro_y1, gyro_z1 = generate_imu_data(
    base_freq=1.5,
    phase_shift=0,
    amplitude_modifier=1.0
)

# IMU2 (medial ankle) - slightly different phase and amplitude
acc_x2, acc_y2, acc_z2, gyro_x2, gyro_y2, gyro_z2 = generate_imu_data(
    base_freq=1.5,
    phase_shift=np.pi/6,  # 30-degree phase shift
    amplitude_modifier=0.9  # Slightly different amplitude
)

# Create DataFrames for both sensors
data = {
    'lateral_ankle': pd.DataFrame({
        'time': time,
        'acc_x': acc_x1,
        'acc_y': acc_y1,
        'acc_z': acc_z1,
        'gyro_x': gyro_x1,
        'gyro_y': gyro_y1,
        'gyro_z': gyro_z1
    }),
    'medial_ankle': pd.DataFrame({
        'time': time,
        'acc_x': acc_x2,
        'acc_y': acc_y2,
        'acc_z': acc_z2,
        'gyro_x': gyro_x2,
        'gyro_y': gyro_y2,
        'gyro_z': gyro_z2
    })
}

# Save to CSV
for sensor_name, df in data.items():
    df.to_csv(f'synthetic_gait_data_{sensor_name}.csv', index=False)

# Plot the data
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot accelerometer data for both sensors
axes[0,0].plot(time, acc_x1, label='Lateral Acc X')
axes[0,0].plot(time, acc_y1, label='Lateral Acc Y')
axes[0,0].plot(time, acc_z1, label='Lateral Acc Z')
axes[0,0].set_title('Lateral Ankle Accelerometer Data')
axes[0,0].set_xlabel('Time (s)')
axes[0,0].set_ylabel('Acceleration (g)')
axes[0,0].legend()

axes[0,1].plot(time, acc_x2, label='Medial Acc X')
axes[0,1].plot(time, acc_y2, label='Medial Acc Y')
axes[0,1].plot(time, acc_z2, label='Medial Acc Z')
axes[0,1].set_title('Medial Ankle Accelerometer Data')
axes[0,1].set_xlabel('Time (s)')
axes[0,1].set_ylabel('Acceleration (g)')
axes[0,1].legend()

# Plot gyroscope data for both sensors
axes[1,0].plot(time, gyro_x1, label='Lateral Gyro X')
axes[1,0].plot(time, gyro_y1, label='Lateral Gyro Y')
axes[1,0].plot(time, gyro_z1, label='Lateral Gyro Z')
axes[1,0].set_title('Lateral Ankle Gyroscope Data')
axes[1,0].set_xlabel('Time (s)')
axes[1,0].set_ylabel('Angular Velocity (dps)')
axes[1,0].legend()

axes[1,1].plot(time, gyro_x2, label='Medial Gyro X')
axes[1,1].plot(time, gyro_y2, label='Medial Gyro Y')
axes[1,1].plot(time, gyro_z2, label='Medial Gyro Z')
axes[1,1].set_title('Medial Ankle Gyroscope Data')
axes[1,1].set_xlabel('Time (s)')
axes[1,1].set_ylabel('Angular Velocity (dps)')
axes[1,1].legend()

plt.tight_layout()
plt.show()