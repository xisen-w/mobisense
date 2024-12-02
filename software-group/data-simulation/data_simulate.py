import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
sampling_rate = 100  # Hz
duration = 10  # seconds
time = np.linspace(0, duration, duration * sampling_rate)

# Simulate accelerometer data (g)
# Assuming a simple sinusoidal pattern to represent walking acceleration
acc_x = 0.5 * np.sin(2 * np.pi * 1.5 * time)  # 1.5 Hz walking frequency
acc_y = 0.1 * np.sin(2 * np.pi * 3 * time)    # 3 Hz lateral sway
acc_z = 1.0 + 0.2 * np.sin(2 * np.pi * 1.5 * time)  # Gravity + vertical oscillation

# Simulate gyroscope data (degrees per second)
gyro_x = 5 * np.sin(2 * np.pi * 1.5 * time)  # Pitch
gyro_y = 2 * np.sin(2 * np.pi * 1.5 * time)  # Roll
gyro_z = 3 * np.sin(2 * np.pi * 1.5 * time)  # Yaw

# Create a DataFrame
data = pd.DataFrame({
    'time': time,
    'acc_x': acc_x,
    'acc_y': acc_y,
    'acc_z': acc_z,
    'gyro_x': gyro_x,
    'gyro_y': gyro_y,
    'gyro_z': gyro_z
})

# Save to CSV
data.to_csv('synthetic_gait_data.csv', index=False)

# Plot the data
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(time, acc_x, label='Acc X')
plt.plot(time, acc_y, label='Acc Y')
plt.plot(time, acc_z, label='Acc Z')
plt.title('Simulated Accelerometer Data')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (g)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, gyro_x, label='Gyro X')
plt.plot(time, gyro_y, label='Gyro Y')
plt.plot(time, gyro_z, label='Gyro Z')
plt.title('Simulated Gyroscope Data')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (dps)')
plt.legend()

plt.tight_layout()
plt.show()