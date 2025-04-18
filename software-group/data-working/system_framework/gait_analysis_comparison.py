# gait_comparison_oop.py

import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks, correlate, welch, detrend
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt


# === Shared Utility Functions ===
def load_trc_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    marker_names = lines[3].strip().split()
    coord_headers = lines[4].strip().split()
    column_names = ['Frame', 'Time']
    marker_names = marker_names[2:]

    for marker in marker_names:
        column_names.extend([f"{marker}_X", f"{marker}_Y", f"{marker}_Z"])

    df = pd.read_csv(path, sep='\s+', skiprows=5, header=None)
    df.columns = column_names[:df.shape[1]]
    return df

def smooth_signal(arr, window_size=11):
    return uniform_filter1d(arr, size=window_size, mode='nearest')

def detect_heel_strikes(time, z_vals, distance=20, prominence=0.001):
    peaks, _ = find_peaks(-z_vals, distance=distance, prominence=prominence)
    return time[peaks], peaks

def detect_imu_toe_offs(time, signal, heel_strike_times):
    toe_off_times = []
    for i in range(len(heel_strike_times) - 1):
        t_start = heel_strike_times[i]
        t_end = heel_strike_times[i + 1]
        mask = (time >= t_start) & (time < t_end)
        segment = signal[mask]
        segment_time = time[mask]
        if len(segment_time) == 0:
            continue
        max_idx = np.argmax(segment)
        toe_off_times.append(segment_time[max_idx])
    return toe_off_times

def compute_stride_metrics(heel_strike_times):
    stride_times = np.diff(heel_strike_times)
    mean_stride_time = np.mean(stride_times)
    stride_variability = np.std(stride_times) / mean_stride_time * 100
    return mean_stride_time, stride_variability, stride_times

def compute_stance_ratio(time, toe_z, heel_strike_times):
    toe_off_times = []
    for i in range(len(heel_strike_times) - 1):
        t_start = heel_strike_times[i]
        t_end = heel_strike_times[i + 1]
        mask = (time >= t_start) & (time < t_end)
        segment = toe_z[mask]
        segment_time = time[mask]
        if len(segment) == 0:
            continue
        min_idx = segment.argmin()
        toe_off_times.append(segment_time[min_idx])
    stance_durations = [toe - heel for heel, toe in zip(heel_strike_times[:-1], toe_off_times)]
    stride_durations = np.diff(heel_strike_times[:len(toe_off_times)])
    stance_ratios = [stance / stride for stance, stride in zip(stance_durations, stride_durations)]
    return np.mean(stance_ratios) * 100, np.array(toe_off_times)

def calc_step_regularities(signal, sampling_rate):
    signal = detrend(signal)
    signal = (signal - np.mean(signal)) / np.std(signal)
    autocorr = correlate(signal, signal, mode='full')
    mid = len(autocorr) // 2
    autocorr = autocorr[mid:]
    autocorr /= np.max(autocorr)
    peaks, _ = find_peaks(autocorr, distance=int(sampling_rate * 0.4))
    if len(peaks) > 1:
        return float(autocorr[peaks[1]])
    return 0.0

def calc_walking_frequency(signal, sampling_rate):
    f, Pxx = welch(signal, fs=sampling_rate, nperseg=256)
    peak_idx = np.argmax(Pxx)
    return f[peak_idx]

def calc_early_heel_rise_ratio(times, sagittal_angles, heel_strikes, toe_offs, threshold=15):
    early_rise_count = 0
    valid_strides = 0
    toe_offs = np.array(toe_offs)
    for hs in heel_strikes:
        future_toe_offs = toe_offs[toe_offs > hs]
        if len(future_toe_offs) == 0:
            continue
        to = future_toe_offs[0]
        mask = (times >= hs) & (times < to)
        if np.mean(sagittal_angles[mask]) > threshold:
            early_rise_count += 1
        valid_strides += 1
    return early_rise_count / valid_strides if valid_strides > 0 else 0.0


# === IMU Analyzer ===
class IMUAnalyzer:
    def __init__(self, imu_path, imu_start_time):
        self.path = imu_path
        self.start_time = imu_start_time
        self.metrics = {}

    def process(self):
        df = pd.read_csv(self.path)
        df = df[df['time'] >= 0.0].reset_index(drop=True)
        time = df['time'].values
        imu_mask = time >= self.start_time
        time = time[imu_mask]

        acc_z = smooth_signal(df['imu0_acc_z'].values[imu_mask])
        acc_x = smooth_signal(df['imu1_acc_x'].values[imu_mask])
        gyro_mag = np.linalg.norm(df[["imu1_gyro_x", "imu1_gyro_y", "imu1_gyro_z"]].values[imu_mask], axis=1)
        gyro_mag = smooth_signal(gyro_mag)

        foot_roll = df['imu1_pitch'].values[imu_mask]
        shank_roll = df['imu0_pitch'].values[imu_mask]
        frontal_angle = foot_roll - shank_roll

        heel_strike_times, _ = detect_heel_strikes(time, acc_z, distance=50, prominence=15)
        mean_stride_time, stride_variability, stride_times = compute_stride_metrics(heel_strike_times)
        stride_count = len(heel_strike_times) - 1
        duration = time[-1] - time[0]
        cadence = (len(heel_strike_times) / duration) * 60 if duration > 0 else 0

        toe_off_times = detect_imu_toe_offs(time, acc_z, heel_strike_times)
        stance_durations = [toe - heel for heel, toe in zip(heel_strike_times[:-1], toe_off_times)]
        stride_durations = np.diff(heel_strike_times[:len(toe_off_times)])
        stance_ratios = [stance / stride for stance, stride in zip(stance_durations, stride_durations)]
        stance_phase_ratio = np.mean(stance_ratios) * 100

        # Stride length/velocity estimation from acc_x
        stride_lengths = []
        stride_times_for_velocity = []
        for i in range(0, len(heel_strike_times) - 2, 2):
            t0 = heel_strike_times[i]
            t1 = heel_strike_times[i + 2]
            idx0 = np.argmin(np.abs(time - t0))
            idx1 = np.argmin(np.abs(time - t1))
            segment = detrend(acc_x[idx0:idx1])
            velocity = detrend(np.cumsum(segment) / 100.0)
            position = np.cumsum(velocity) / 100.0
            stride_lengths.append(np.max(position) - np.min(position))
            stride_times_for_velocity.append(t1 - t0)
        mean_stride_length = np.mean(stride_lengths) if stride_lengths else 0
        mean_stride_velocity = np.mean([l / t for l, t in zip(stride_lengths, stride_times_for_velocity) if t > 0])

        # Frontal angle metrics
        peak_inv = []
        inv_var = []
        for i in range(len(heel_strike_times) - 1):
            mask = (time >= heel_strike_times[i]) & (time < heel_strike_times[i + 1])
            segment = frontal_angle[mask]
            if len(segment):
                peak_inv.append(np.max(segment))
                inv_var.append(np.std(segment))

        # Advanced metrics
        sagittal_angle = np.zeros_like(time)  # placeholder
        advanced = {
            "Step Regularity": calc_step_regularities(acc_z, 100),
            "Walking Frequency (Hz)": calc_walking_frequency(acc_z, 100),
            "Early Heel Rise Ratio": calc_early_heel_rise_ratio(time, sagittal_angle, heel_strike_times, toe_off_times)
        }

        self.metrics = {
            "Stride Count": stride_count,
            "Cadence": cadence,
            "Mean Stride Time": mean_stride_time,
            "Stride Time Variability": stride_variability,
            "Stance Phase Ratio": stance_phase_ratio,
            "Mean Stride Length": mean_stride_length,
            "Mean Stride Velocity": mean_stride_velocity,
            "Mean Peak Inversion": np.mean(peak_inv) if peak_inv else 0,
            "Inversion Variability": np.mean(inv_var) if inv_var else 0,
            **advanced
        }

    def get_metrics(self):
        return self.metrics


# === OpenCap Analyzer ===
class OpenCapAnalyzer:
    def __init__(self, trc_path, start_time):
        self.path = trc_path
        self.start_time = start_time
        self.metrics = {}

    def process(self):
        df = load_trc_file(self.path)
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
        df = df[df["Time"] >= self.start_time].reset_index(drop=True)

        time = df["Time"].values
        heel_z = df["RHeel_Z"].values
        toe_z = df["RBigToe_Z"].values
        heel_x = df["RHeel_X"].values
        heel_y = df["RHeel_Y"].values
        toe_y = df["RBigToe_Y"].values

        heel_strike_times, peak_idxs = detect_heel_strikes(time, heel_z)
        mean_stride_time, stride_variability, stride_times = compute_stride_metrics(heel_strike_times)
        duration = time[-1] - time[0]
        stride_count = len(heel_strike_times) - 1
        cadence = (len(heel_strike_times) / duration) * 60 if duration > 0 else 0
        stance_phase_ratio, toe_offs = compute_stance_ratio(time, toe_z, heel_strike_times)

        stride_lengths = [abs(heel_x[i+1] - heel_x[i]) for i in range(len(peak_idxs) - 1)]
        stride_velocities = [l / (time[peak_idxs[i+1]] - time[peak_idxs[i]]) for i, l in enumerate(stride_lengths)]
        mean_stride_length = np.mean(stride_lengths) if stride_lengths else 0
        mean_stride_velocity = np.mean(stride_velocities) if stride_velocities else 0

        foot_vec_yz = np.vstack((toe_y - heel_y, toe_z - heel_z)).T
        foot_roll_angle = np.rad2deg(np.arctan2(foot_vec_yz[:, 0], foot_vec_yz[:, 1]))
        foot_roll_angle = smooth_signal(foot_roll_angle)
        peak_inv, inv_var = [], []
        for i in range(len(heel_strike_times) - 1):
            mask = (time >= heel_strike_times[i]) & (time < heel_strike_times[i + 1])
            segment = foot_roll_angle[mask]
            if len(segment):
                peak_inv.append(np.max(segment))
                inv_var.append(np.std(segment))

        sagittal_angle = np.zeros_like(time)
        advanced = {
            "Step Regularity": calc_step_regularities(heel_z, 100),
            "Walking Frequency (Hz)": calc_walking_frequency(heel_z, 100),
            "Early Heel Rise Ratio": calc_early_heel_rise_ratio(time, sagittal_angle, heel_strike_times, toe_offs)
        }

        self.metrics = {
            "Stride Count": stride_count,
            "Cadence": cadence,
            "Mean Stride Time": mean_stride_time,
            "Stride Time Variability": stride_variability,
            "Stance Phase Ratio": stance_phase_ratio,
            "Mean Stride Length": mean_stride_length,
            "Mean Stride Velocity": mean_stride_velocity,
            "Mean Peak Inversion": np.mean(peak_inv) if peak_inv else 0,
            "Inversion Variability": np.mean(inv_var) if inv_var else 0,
            **advanced
        }

    def get_metrics(self):
        return self.metrics


# === Comparison Controller ===
class GaitComparisonAnalyzer:
    def __init__(self, imu_path, trc_path, imu_start, trc_start):
        self.imu_analyzer = IMUAnalyzer(imu_path, imu_start)
        self.opencap_analyzer = OpenCapAnalyzer(trc_path, trc_start)

    def process(self):
        self.imu_analyzer.process()
        self.opencap_analyzer.process()

    def compare(self):
        imu_metrics = self.imu_analyzer.get_metrics()
        oc_metrics = self.opencap_analyzer.get_metrics()
        all_keys = sorted(set(imu_metrics) | set(oc_metrics))
        return {k: (oc_metrics.get(k), imu_metrics.get(k)) for k in all_keys}
