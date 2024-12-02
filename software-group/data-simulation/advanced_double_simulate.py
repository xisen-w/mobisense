import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
sampling_rate = 100  # Hz
duration = 10  # seconds
time = np.linspace(0, duration, duration * sampling_rate)
noise_std_acc = 0.02  # Standard deviation for accelerometer noise
noise_std_gyro = 0.05  # Standard deviation for gyroscope noise

def generate_imu_data_with_angles(base_freq=1.5, phase_shift=0, amplitude_modifier=1.0):
    """Generate IMU data derived from angular motions."""
    # Angular displacement for dorsiflexion (~10°) and plantarflexion (~20°)
    angles = amplitude_modifier * (10 + 10 * np.sin(2 * np.pi * base_freq * time + phase_shift))  # Degrees
    
    # Convert angles to radians for calculations
    angles_rad = np.radians(angles)
    
    # Angular velocity (gyroscope data) - first derivative of angle
    angular_velocity = np.gradient(angles_rad, 1 / sampling_rate)  # rad/s
    
    # Linear acceleration (accelerometer data) - second derivative of angle
    linear_acc = np.gradient(angular_velocity, 1 / sampling_rate)  # rad/s²
    
    # Convert to accelerometer and gyroscope readings
    acc_x = linear_acc * amplitude_modifier + np.random.normal(0, noise_std_acc, size=time.shape)
    acc_y = 0.1 * linear_acc * amplitude_modifier + np.random.normal(0, noise_std_acc, size=time.shape)
    acc_z = 1.0 + 0.2 * linear_acc * amplitude_modifier + np.random.normal(0, noise_std_acc, size=time.shape)
    
    gyro_x = angular_velocity * amplitude_modifier + np.random.normal(0, noise_std_gyro, size=time.shape)
    gyro_y = 0.5 * angular_velocity * amplitude_modifier + np.random.normal(0, noise_std_gyro, size=time.shape)
    gyro_z = 0.7 * angular_velocity * amplitude_modifier + np.random.normal(0, noise_std_gyro, size=time.shape)
    
    return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, angles

# Generate data for two IMUs with differences
acc_x1, acc_y1, acc_z1, gyro_x1, gyro_y1, gyro_z1, angles1 = generate_imu_data_with_angles(
    base_freq=1.5,
    phase_shift=0,
    amplitude_modifier=1.0
)

acc_x2, acc_y2, acc_z2, gyro_x2, gyro_y2, gyro_z2, angles2 = generate_imu_data_with_angles(
    base_freq=1.5,
    phase_shift=np.pi / 6,  # 30-degree phase shift
    amplitude_modifier=0.9
)

# Create DataFrames for both sensors
data = {
    'lateral_ankle': pd.DataFrame({
        'time': time,
        'angle': angles1,
        'acc_x': acc_x1,
        'acc_y': acc_y1,
        'acc_z': acc_z1,
        'gyro_x': gyro_x1,
        'gyro_y': gyro_y1,
        'gyro_z': gyro_z1
    }),
    'medial_ankle': pd.DataFrame({
        'time': time,
        'angle': angles2,
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
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

# Plot angular data
axes[0, 0].plot(time, angles1, label='Lateral Angle')
axes[0, 0].set_title('Lateral Ankle Angular Motion (Degrees)')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Angle (Degrees)')
axes[0, 0].legend()

axes[0, 1].plot(time, angles2, label='Medial Angle')
axes[0, 1].set_title('Medial Ankle Angular Motion (Degrees)')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Angle (Degrees)')
axes[0, 1].legend()

# Plot accelerometer data for both sensors
axes[1, 0].plot(time, acc_x1, label='Lateral Acc X')
axes[1, 0].plot(time, acc_y1, label='Lateral Acc Y')
axes[1, 0].plot(time, acc_z1, label='Lateral Acc Z')
axes[1, 0].set_title('Lateral Ankle Accelerometer Data')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Acceleration (g)')
axes[1, 0].legend()

axes[1, 1].plot(time, acc_x2, label='Medial Acc X')
axes[1, 1].plot(time, acc_y2, label='Medial Acc Y')
axes[1, 1].plot(time, acc_z2, label='Medial Acc Z')
axes[1, 1].set_title('Medial Ankle Accelerometer Data')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Acceleration (g)')
axes[1, 1].legend()

# Plot gyroscope data for both sensors
axes[2, 0].plot(time, gyro_x1, label='Lateral Gyro X')
axes[2, 0].plot(time, gyro_y1, label='Lateral Gyro Y')
axes[2, 0].plot(time, gyro_z1, label='Lateral Gyro Z')
axes[2, 0].set_title('Lateral Ankle Gyroscope Data')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].set_ylabel('Angular Velocity (rad/s)')
axes[2, 0].legend()

axes[2, 1].plot(time, gyro_x2, label='Medial Gyro X')
axes[2, 1].plot(time, gyro_y2, label='Medial Gyro Y')
axes[2, 1].plot(time, gyro_z2, label='Medial Gyro Z')
axes[2, 1].set_title('Medial Ankle Gyroscope Data')
axes[2, 1].set_xlabel('Time (s)')
axes[2, 1].set_ylabel('Angular Velocity (rad/s)')
axes[2, 1].legend()

plt.tight_layout()
plt.show()