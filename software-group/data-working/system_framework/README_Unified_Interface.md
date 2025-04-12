# MobiSense Unified Interface

This document provides a comprehensive guide to the MobiSense Unified Interface - a Streamlit-based web application for experiment setup, data management, and visualization.

## Overview

The MobiSense Unified Interface provides researchers and clinicians with a user-friendly environment for:

1. Uploading and registering IMU data experiments
2. Visualizing sensor data (acceleration, gyroscope, orientation angles, and dorsiflexion)
3. Managing participant information and experiment metadata
4. Analyzing and comparing experiment results
5. Performing detailed gait and biomechanical analysis

## System Architecture

The interface consists of four main components:

1. **Data Upload & Experiment Setup**: For registering new experiments with participant data
2. **Experiment Viewer**: For reviewing registered experiments and their metadata
3. **Data Visualization**: For interactive visualization of IMU and angle data
4. **Gait Analytics**: For comprehensive gait analysis and biomechanical assessment

## Features

### Data Upload & Experiment Setup

- CSV file upload with preview
- Participant information registration
  - Required: ID, height, weight, age, gender
  - Optional: stride length, stride number per minute, injury date/type
- Experiment metadata recording
- Automatic data validation and angle detection
- Raw data preview with column detection

### Experiment Viewer

- Expandable experiment cards for easy browsing
- Detailed participant information display
- Metadata visualization
- Data preview functionality
- Column inspection and validation

### Data Visualization

- Interactive IMU data plots:
  - Acceleration (X, Y, Z axes)
  - Gyroscope (X, Y, Z axes)
  - Orientation angles (Roll, Pitch, Yaw)
  - Dorsiflexion angle (when available)
- Sample range selection for detailed analysis
- IMU selection for multi-sensor experiments
- Data statistics and debugging tools

### Gait Analytics Dashboard

- Comprehensive gait analysis from IMU data:
  - Ankle joint angle calculation and visualization
  - Range of motion (ROM) metrics
  - Gait event detection (heel strikes and toe-offs)
  - Gait parameter calculation
  - Ankle stability assessment
  - Gait pattern classification
- Interactive sample range selection for targeted analysis
- Detailed visualization components:
  - Ankle joint angle plots (sagittal and frontal planes)
  - Gait event detection visualization
  - Stylized metric tables
  - Risk assessment gauges with color-coding
  - Gait pattern analysis with anomaly detection
- Clinical insights:
  - Calculation of dorsiflexion/plantarflexion range
  - Inversion/eversion assessment
  - Sprain risk analysis based on ankle kinematics
  - Gait abnormality scoring and identification

## Data Structure

The system is designed to work with IMU data in CSV format with the following structure:

### Required Columns
- `imu0_acc_x`, `imu0_acc_y`, `imu0_acc_z`: Accelerometer data for IMU 0
- `imu0_gyro_x`, `imu0_gyro_y`, `imu0_gyro_z`: Gyroscope data for IMU 0
- `imu0_timestamp`: Timestamp for IMU 0 measurements

### Optional Columns
- `imu0_roll`, `imu0_pitch`, `imu0_yaw`: Orientation angles for IMU 0
- `imu1_acc_x`, `imu1_acc_y`, `imu1_acc_z`: Accelerometer data for IMU 1
- `imu1_gyro_x`, `imu1_gyro_y`, `imu1_gyro_z`: Gyroscope data for IMU 1
- `imu1_roll`, `imu1_pitch`, `imu1_yaw`: Orientation angles for IMU 1
- `imu1_timestamp`: Timestamp for IMU 1 measurements
- `dorsiflexion_angle`: Calculated dorsiflexion angle between IMUs

## Installation and Setup

### Prerequisites
- Python 3.6+
- pip package manager

### Setup Instructions

1. Clone the repository or download the source code
2. Navigate to the system_framework directory
3. Run the setup script:
   ```bash
   ./setup.sh
   ```
4. Start the application:
   ```bash
   ./run_app.sh
   ```

### Manual Setup

If you prefer to set up manually:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the application:
   ```bash
   streamlit run streamlit_app.py
   ```

## Experiment Workflow

1. **Prepare Data**: Ensure your CSV file contains the required columns
2. **Upload Data**: Use the Data Upload tab to upload your CSV file
3. **Enter Participant Info**: Fill in participant details
4. **Register Experiment**: Submit the form to register the experiment
5. **View Experiment**: Navigate to the View Experiments tab to see your registered experiment
6. **Visualize Data**: Use the Data Visualization tab to explore the sensor data
7. **Analyze Gait**: Use the Gait Analytics tab to perform detailed gait analysis

## Analytics Modules

The MobiSense system incorporates several analytical modules:

### Range of Motion (ROM) Analysis

The ROM analysis module (from RoM_updated.py) provides:
- Calculation of ankle angles in multiple planes (sagittal, frontal, transverse)
- Range of motion metrics including maximum dorsiflexion, plantarflexion, inversion, and eversion
- Complementary filtering of IMU data for accurate angle estimation
- Direct use of measured angles when available or calculation from raw IMU data when not

### Gait Analysis

The gait analysis module (from gait_analysis.py) offers:
- Gait event detection using acceleration patterns
- Calculation of temporal gait parameters (cadence, stride time, stance phase ratio)
- Pathological gait assessment
- Stride parameter calculation
- Gait symmetry analysis

### Sprain Risk Assessment

The sprain risk assessment functionality provides:
- Angular velocity analysis for rapid inversion/eversion movements
- Stability index calculation
- Sudden inversion event detection
- Visual risk representation using gauge charts

## Working with the Analytics Dashboard

The Gait Analytics dashboard is designed for clinical assessment and research:

1. **Data Selection**: Choose an experiment and select a sample range for analysis
2. **Angle Calculation**: The system automatically calculates ankle angles using either provided angles or IMU data
3. **Gait Event Detection**: Heel strikes and toe-offs are automatically detected from acceleration patterns
4. **Parameter Review**: Review the calculated metrics in the tables and visualizations
5. **Risk Assessment**: Examine the risk gauges for ankle instability indicators
6. **Pattern Analysis**: Review the gait pattern classification for potential abnormalities

## File Organization

- `streamlit_app.py`: Main Streamlit application
- `experiments.py`: Classes for experiment and participant management
- `RoM.py` and `RoM_updated.py`: Range of motion calculation modules
- `gait_analysis.py`: Gait parameter calculation and analysis module
- `setup.sh`: Setup script for environment preparation
- `run_app.sh`: Script to run the application
- `experiments/`: Directory where experiment data and metadata are stored

## Usage Notes

- If angle data is not detected when it should be, use the debug tools to investigate
- The system automatically attempts to recover angle data if it exists in the raw CSV but isn't properly parsed
- Orientation angles (roll, pitch, yaw) are visualized separately from acceleration and gyroscope data
- Expandable sections in the interface provide additional details and debugging information when needed

## Interpretation of Analytics Results

When using the Gait Analytics dashboard, consider these guidelines:

- **Range of Motion**: Normal ankle ROM is approximately 20° dorsiflexion and 50° plantarflexion. Significant deviations may indicate mobility issues.
- **Gait Parameters**: Typical cadence ranges from 90-120 steps/minute. Increased stride variability (>10%) may indicate instability.
- **Risk Factors**: Max inversion velocity >200°/s or stability index >50 may indicate increased sprain risk.
- **Pattern Classification**: The system categorizes gait as normal, mildly abnormal, or significantly abnormal based on a composite score of metrics.

## Technical Notes

- The analytics modules use a default sampling rate of 100Hz. If your data uses a different rate, adjust the initialization parameters.
- Gait event detection works best with walking or running data; standing or irregular movements may cause false detections.
- The system prefers to use direct angle measurements when available but can calculate them from IMU data when necessary.
- For best results, place IMU 0 on the shank and IMU 1 on the foot. 