import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
sampling_rate = 100  # Hz
duration = 10  # seconds
time = np.linspace(0, duration, duration * sampling_rate)
noise_std_acc = 0.02  # Standard deviation for accelerometer noise
noise_std_gyro = 0.05  # Standard deviation for gyroscope noise

def generate_sprained_ankle_data(dorsiflexion_rom, plantarflexion_rom, noise_modifier=1.0, phase_shift=0):
    """Generate IMU data derived from angular motions for sprained ankle simulation."""
    # Angular displacement for dorsiflexion and plantarflexion
    angles = (dorsiflexion_rom + (plantarflexion_rom - dorsiflexion_rom) *
              np.sin(2 * np.pi * 1.5 * time + phase_shift))  # Degrees
    
    # Convert angles to radians for calculations
    angles_rad = np.radians(angles)
    
    # Angular velocity (gyroscope data) - first derivative of angle
    angular_velocity = np.gradient(angles_rad, 1 / sampling_rate)  # rad/s
    
    # Linear acceleration (accelerometer data) - second derivative of angle
    linear_acc = np.gradient(angular_velocity, 1 / sampling_rate)  # rad/s²
    
    # Add noise to simulate sensor variability
    acc_x = linear_acc + np.random.normal(0, noise_std_acc * noise_modifier, size=time.shape)
    acc_y = 0.1 * linear_acc + np.random.normal(0, noise_std_acc * noise_modifier, size=time.shape)
    acc_z = 1.0 + 0.2 * linear_acc + np.random.normal(0, noise_std_acc * noise_modifier, size=time.shape)
    
    gyro_x = angular_velocity + np.random.normal(0, noise_std_gyro * noise_modifier, size=time.shape)
    gyro_y = 0.5 * angular_velocity + np.random.normal(0, noise_std_gyro * noise_modifier, size=time.shape)
    gyro_z = 0.7 * angular_velocity + np.random.normal(0, noise_std_gyro * noise_modifier, size=time.shape)
    
    return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, angles

# Generate data for injured and non-injured sides
injured_acc_x, injured_acc_y, injured_acc_z, injured_gyro_x, injured_gyro_y, injured_gyro_z, injured_angles = \
    generate_sprained_ankle_data(
        dorsiflexion_rom=8.1,
        plantarflexion_rom=47.8,
        noise_modifier=1.2,
        phase_shift=0
    )

non_injured_acc_x, non_injured_acc_y, non_injured_acc_z, non_injured_gyro_x, non_injured_gyro_y, non_injured_gyro_z, non_injured_angles = \
    generate_sprained_ankle_data(
        dorsiflexion_rom=11.5,
        plantarflexion_rom=52.3,
        noise_modifier=1.0,
        phase_shift=np.pi / 6
    )

# Create DataFrames for both sides
data = {
    'injured_side': pd.DataFrame({
        'time': time,
        'angle': injured_angles,
        'acc_x': injured_acc_x,
        'acc_y': injured_acc_y,
        'acc_z': injured_acc_z,
        'gyro_x': injured_gyro_x,
        'gyro_y': injured_gyro_y,
        'gyro_z': injured_gyro_z
    }),
    'non_injured_side': pd.DataFrame({
        'time': time,
        'angle': non_injured_angles,
        'acc_x': non_injured_acc_x,
        'acc_y': non_injured_acc_y,
        'acc_z': non_injured_acc_z,
        'gyro_x': non_injured_gyro_x,
        'gyro_y': non_injured_gyro_y,
        'gyro_z': non_injured_gyro_z
    })
}

# Save to CSV
for side, df in data.items():
    df.to_csv(f'synthetic_gait_data_{side}.csv', index=False)

# Plot the data
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

# Plot angular data
axes[0, 0].plot(time, injured_angles, label='Injured Angle')
axes[0, 0].set_title('Injured Side Angular Motion (Degrees)')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Angle (Degrees)')
axes[0, 0].legend()

axes[0, 1].plot(time, non_injured_angles, label='Non-Injured Angle')
axes[0, 1].set_title('Non-Injured Side Angular Motion (Degrees)')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Angle (Degrees)')
axes[0, 1].legend()

# Plot accelerometer data
axes[1, 0].plot(time, injured_acc_x, label='Injured Acc X')
axes[1, 0].plot(time, injured_acc_y, label='Injured Acc Y')
axes[1, 0].plot(time, injured_acc_z, label='Injured Acc Z')
axes[1, 0].set_title('Injured Side Accelerometer Data')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Acceleration (g)')
axes[1, 0].legend()

axes[1, 1].plot(time, non_injured_acc_x, label='Non-Injured Acc X')
axes[1, 1].plot(time, non_injured_acc_y, label='Non-Injured Acc Y')
axes[1, 1].plot(time, non_injured_acc_z, label='Non-Injured Acc Z')
axes[1, 1].set_title('Non-Injured Side Accelerometer Data')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Acceleration (g)')
axes[1, 1].legend()

# Plot gyroscope data
axes[2, 0].plot(time, injured_gyro_x, label='Injured Gyro X')
axes[2, 0].plot(time, injured_gyro_y, label='Injured Gyro Y')
axes[2, 0].plot(time, injured_gyro_z, label='Injured Gyro Z')
axes[2, 0].set_title('Injured Side Gyroscope Data')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].set_ylabel('Angular Velocity (rad/s)')
axes[2, 0].legend()

axes[2, 1].plot(time, non_injured_gyro_x, label='Non-Injured Gyro X')
axes[2, 1].plot(time, non_injured_gyro_y, label='Non-Injured Gyro Y')
axes[2, 1].plot(time, non_injured_gyro_z, label='Non-Injured Gyro Z')
axes[2, 1].set_title('Non-Injured Side Gyroscope Data')
axes[2, 1].set_xlabel('Time (s)')
axes[2, 1].set_ylabel('Angular Velocity (rad/s)')
axes[2, 1].legend()

plt.tight_layout()
plt.show()