import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
sampling_rate = 100  # Hz
duration = 10  # seconds
time = np.linspace(0, duration, duration * sampling_rate)

def generate_imu_data(base_freq=1.5, phase_shift=0, amplitude_modifier=1.0, noise_level=0.01):
    """
    Generate IMU data simulating different recovery stages of ankle sprain
    - Lower base_freq indicates more cautious/slower walking
    - Lower amplitude_modifier indicates reduced range of motion
    - Higher noise_level indicates less stable movement
    """
    # Accelerometer data (g)
    # Reduced amplitude in acc_y indicates limited inversion/eversion
    acc_x = amplitude_modifier * 0.5 * np.sin(2 * np.pi * base_freq * time + phase_shift)
    acc_y = amplitude_modifier * 0.1 * np.sin(2 * np.pi * 3 * time + phase_shift)
    acc_z = 1.0 + amplitude_modifier * 0.2 * np.sin(2 * np.pi * base_freq * time + phase_shift)
    
    # Gyroscope data (degrees per second)
    # Reduced amplitude in gyro_y indicates limited dorsiflexion/plantarflexion
    gyro_x = amplitude_modifier * 5 * np.sin(2 * np.pi * base_freq * time + phase_shift)
    gyro_y = amplitude_modifier * 2 * np.sin(2 * np.pi * base_freq * time + phase_shift)
    gyro_z = amplitude_modifier * 3 * np.sin(2 * np.pi * base_freq * time + phase_shift)
    
    # Add Gaussian noise
    acc_x += np.random.normal(0, noise_level, size=acc_x.shape)
    acc_y += np.random.normal(0, noise_level, size=acc_y.shape)
    acc_z += np.random.normal(0, noise_level, size=acc_z.shape)
    gyro_x += np.random.normal(0, noise_level, size=gyro_x.shape)
    gyro_y += np.random.normal(0, noise_level, size=gyro_y.shape)
    gyro_z += np.random.normal(0, noise_level, size=gyro_z.shape)
    
    return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

# Recovery stages configuration
recovery_stages = {
    'Grade_III_Initial': {  # Most severe stage
        'base_freq': 1.0,           # Slowest walking
        'amplitude_modifier': 0.4,   # Most limited ROM
        'noise_level': 0.04         # Most unstable
    },
    'Grade_III_Late': {
        'base_freq': 1.1,
        'amplitude_modifier': 0.5,
        'noise_level': 0.035
    },
    'Grade_II_Initial': {
        'base_freq': 1.2,
        'amplitude_modifier': 0.6,
        'noise_level': 0.03
    },
    'Grade_II_Late': {
        'base_freq': 1.3,
        'amplitude_modifier': 0.7,
        'noise_level': 0.025
    },
    'Grade_I_Initial': {
        'base_freq': 1.4,
        'amplitude_modifier': 0.8,
        'noise_level': 0.02
    },
    'Grade_I_Late': {  # Almost recovered
        'base_freq': 1.5,           # Normal walking speed
        'amplitude_modifier': 0.9,   # Almost normal ROM
        'noise_level': 0.015        # More stable movement
    }
}

# Generate and save data for each recovery stage
for stage, params in recovery_stages.items():
    # Generate data for lateral ankle sensor
    acc_x1, acc_y1, acc_z1, gyro_x1, gyro_y1, gyro_z1 = generate_imu_data(
        base_freq=params['base_freq'],
        amplitude_modifier=params['amplitude_modifier'],
        noise_level=params['noise_level']
    )
    
    # Generate data for medial ankle sensor (slightly different phase and amplitude)
    acc_x2, acc_y2, acc_z2, gyro_x2, gyro_y2, gyro_z2 = generate_imu_data(
        base_freq=params['base_freq'],
        phase_shift=np.pi/6,  # 30-degree phase shift
        amplitude_modifier=params['amplitude_modifier'] * 0.9,  # Slightly different amplitude
        noise_level=params['noise_level']
    )
    
    # Create DataFrames
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
        df.to_csv(f'synthetic_gait_data_{stage}_{sensor_name}.csv', index=False)
    
    # Plot the data
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'IMU Data for {stage.replace("_", " ")}')
    
    # Plot accelerometer data for both sensors
    axes[0,0].plot(time, acc_x1, label='Acc X')
    axes[0,0].plot(time, acc_y1, label='Acc Y')
    axes[0,0].plot(time, acc_z1, label='Acc Z')
    axes[0,0].set_title('Lateral Ankle Accelerometer')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Acceleration (g)')
    axes[0,0].legend()
    
    axes[0,1].plot(time, acc_x2, label='Acc X')
    axes[0,1].plot(time, acc_y2, label='Acc Y')
    axes[0,1].plot(time, acc_z2, label='Acc Z')
    axes[0,1].set_title('Medial Ankle Accelerometer')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Acceleration (g)')
    axes[0,1].legend()
    
    # Plot gyroscope data for both sensors
    axes[1,0].plot(time, gyro_x1, label='Gyro X')
    axes[1,0].plot(time, gyro_y1, label='Gyro Y')
    axes[1,0].plot(time, gyro_z1, label='Gyro Z')
    axes[1,0].set_title('Lateral Ankle Gyroscope')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Angular Velocity (dps)')
    axes[1,0].legend()
    
    axes[1,1].plot(time, gyro_x2, label='Gyro X')
    axes[1,1].plot(time, gyro_y2, label='Gyro Y')
    axes[1,1].plot(time, gyro_z2, label='Gyro Z')
    axes[1,1].set_title('Medial Ankle Gyroscope')
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('Angular Velocity (dps)')
    axes[1,1].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()