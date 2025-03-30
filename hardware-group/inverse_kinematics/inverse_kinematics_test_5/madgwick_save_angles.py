import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation as R
from scipy.stats import pearsonr
import math

# Constants
csv_file = "/Users/francescobalanzoni/Documents/Python/MEng/3YP/mobisense/software-group/data-working/assets/mar12exp/SyncedData_OpenCapAngle_IMU/walk1_synced.csv"
frequency = 100
start_time = 0.5
end_time = 11
angle_offset = 0
time_offset = 0

gain = 0.001
zeta = 0.01
window_size = 50

class MadgwickWithBias(Madgwick):
    def __init__(self, gain, frequency, zeta):
        super().__init__(gain=gain, frequency=frequency, zeta=zeta)
        self.zeta = zeta
        self.bias = np.zeros(3)

    def updateIMU(self, q, gyr, acc):
        acc_corrected = acc - self.zeta * self.bias
        q[:] = super().updateIMU(q, gyr=gyr, acc=acc_corrected)
        self.bias += self.zeta * (acc - acc_corrected)
        return q

def load_imu_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

def remove_drift(signal, window_size):
    return signal - pd.Series(signal).rolling(window=window_size, center=True, min_periods=1).mean()

def add_madgwick_angle(df, gain, frequency, zeta):
    madgwick = MadgwickWithBias(gain, frequency, zeta)
    
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    madgwick_angles = []
    
    for i in range(len(df)):
        accel = np.array([df.loc[i, 'imu0_acc_x'], df.loc[i, 'imu0_acc_y'], df.loc[i, 'imu0_acc_z']])
        gyro = np.radians(np.array([df.loc[i, 'imu0_gyro_x'], df.loc[i, 'imu0_gyro_y'], df.loc[i, 'imu0_gyro_z']]))
        q = madgwick.updateIMU(q, gyr=gyro, acc=accel)
        r = R.from_quat(q)
        _, pitch, _ = r.as_euler('xyz', degrees=True)
        madgwick_angles.append(pitch)
    
    df['madgwick_angle'] = madgwick_angles
    return df

def main():
    df = load_imu_data(csv_file)
    df = add_madgwick_angle(df, gain, frequency, zeta)
    df.to_csv(csv_file, index=False)
    print("Updated CSV with Madgwick angles saved.")

if __name__ == "__main__":
    main()
