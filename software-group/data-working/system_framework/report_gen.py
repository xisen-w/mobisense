import streamlit as st
import pandas as pd
import numpy as np
from RoM import RangeOfMotionAnalyzer
import matplotlib.pyplot as plt
from io import BytesIO

# Import necessary functions from existing modules
from gait_analysis import GaitAnalyzer
from train_force_prediction_augment import load_and_prepare_sequences, create_simple_model, standardize_data, evaluate_predictions

# Set up Streamlit interface
st.title('IMU Data Analysis and Report Generation')

# Input fields for IMU data and participant data
imu_data_file = st.file_uploader('Upload IMU Data CSV', type='csv')
participant_weight = st.number_input('Participant Weight (kg)', min_value=30, max_value=150, value=70)

if imu_data_file is not None:
    # Load IMU data
    imu_data = pd.read_csv(imu_data_file)
    st.write('IMU Data:', imu_data.head())
    
    # Perform RoM analysis
    st.subheader('Range of Motion Analysis')
    # Create an instance of the RangeOfMotionAnalyzer
    rom_analyzer = RangeOfMotionAnalyzer()
    
    # Calculate ankle angles
    ankle_angles = rom_analyzer.calculate_ankle_angles(imu_data)
    
    # Display ankle angles data
    st.write("Ankle Angles Data:")
    st.write(ankle_angles.head())
    
    # Visualize ankle angles
    st.subheader("Ankle Angles Visualization")
    
    # Create tabs for different angle planes
    angle_tabs = st.tabs(["Sagittal Plane", "Frontal Plane", "Transverse Plane", "All Planes"])
    
    with angle_tabs[0]:
        st.write("### Sagittal Plane (Dorsiflexion/Plantarflexion)")
        fig_sagittal, ax_sagittal = plt.subplots(figsize=(10, 6))
        ax_sagittal.plot(ankle_angles.index, ankle_angles['sagittal_angle'], color='blue')
        ax_sagittal.set_xlabel('Sample')
        ax_sagittal.set_ylabel('Angle (degrees)')
        ax_sagittal.set_title('Sagittal Plane Ankle Angle')
        ax_sagittal.grid(True)
        
        # Add horizontal line at neutral position
        neutral_sagittal = np.median(ankle_angles['sagittal_angle'])
        ax_sagittal.axhline(y=neutral_sagittal, color='r', linestyle='--', label=f'Neutral Position ({neutral_sagittal:.2f}°)')
        
        # Add min and max lines
        max_sagittal = ankle_angles['sagittal_angle'].max()
        min_sagittal = ankle_angles['sagittal_angle'].min()
        ax_sagittal.axhline(y=max_sagittal, color='g', linestyle=':', label=f'Max ({max_sagittal:.2f}°)')
        ax_sagittal.axhline(y=min_sagittal, color='orange', linestyle=':', label=f'Min ({min_sagittal:.2f}°)')
        
        ax_sagittal.legend()
        st.pyplot(fig_sagittal)
        
        # Display ROM metrics for sagittal plane
        st.write(f"**Dorsiflexion Range:** {max_sagittal - neutral_sagittal:.2f}°")
        st.write(f"**Plantarflexion Range:** {neutral_sagittal - min_sagittal:.2f}°")
        st.write(f"**Total ROM:** {max_sagittal - min_sagittal:.2f}°")
    
    with angle_tabs[1]:
        st.write("### Frontal Plane (Inversion/Eversion)")
        fig_frontal, ax_frontal = plt.subplots(figsize=(10, 6))
        ax_frontal.plot(ankle_angles.index, ankle_angles['frontal_angle'], color='green')
        ax_frontal.set_xlabel('Sample')
        ax_frontal.set_ylabel('Angle (degrees)')
        ax_frontal.set_title('Frontal Plane Ankle Angle')
        ax_frontal.grid(True)
        
        # Add horizontal line at neutral position
        neutral_frontal = np.median(ankle_angles['frontal_angle'])
        ax_frontal.axhline(y=neutral_frontal, color='r', linestyle='--', label=f'Neutral Position ({neutral_frontal:.2f}°)')
        
        # Add min and max lines
        max_frontal = ankle_angles['frontal_angle'].max()
        min_frontal = ankle_angles['frontal_angle'].min()
        ax_frontal.axhline(y=max_frontal, color='g', linestyle=':', label=f'Max ({max_frontal:.2f}°)')
        ax_frontal.axhline(y=min_frontal, color='orange', linestyle=':', label=f'Min ({min_frontal:.2f}°)')
        
        ax_frontal.legend()
        st.pyplot(fig_frontal)
        
        # Display ROM metrics for frontal plane
        st.write(f"**Inversion Range:** {max_frontal - neutral_frontal:.2f}°")
        st.write(f"**Eversion Range:** {neutral_frontal - min_frontal:.2f}°")
        st.write(f"**Total ROM:** {max_frontal - min_frontal:.2f}°")
    
    with angle_tabs[2]:
        st.write("### Transverse Plane (Internal/External Rotation)")
        fig_transverse, ax_transverse = plt.subplots(figsize=(10, 6))
        ax_transverse.plot(ankle_angles.index, ankle_angles['transverse_angle'], color='purple')
        ax_transverse.set_xlabel('Sample')
        ax_transverse.set_ylabel('Angle (degrees)')
        ax_transverse.set_title('Transverse Plane Ankle Angle')
        ax_transverse.grid(True)
        
        # Add horizontal line at neutral position
        neutral_transverse = np.median(ankle_angles['transverse_angle'])
        ax_transverse.axhline(y=neutral_transverse, color='r', linestyle='--', label=f'Neutral Position ({neutral_transverse:.2f}°)')
        
        # Add min and max lines
        max_transverse = ankle_angles['transverse_angle'].max()
        min_transverse = ankle_angles['transverse_angle'].min()
        ax_transverse.axhline(y=max_transverse, color='g', linestyle=':', label=f'Max ({max_transverse:.2f}°)')
        ax_transverse.axhline(y=min_transverse, color='orange', linestyle=':', label=f'Min ({min_transverse:.2f}°)')
        
        ax_transverse.legend()
        st.pyplot(fig_transverse)
        
        # Display ROM metrics for transverse plane
        st.write(f"**Internal Rotation Range:** {max_transverse - neutral_transverse:.2f}°")
        st.write(f"**External Rotation Range:** {neutral_transverse - min_transverse:.2f}°")
        st.write(f"**Total ROM:** {max_transverse - min_transverse:.2f}°")
    
    with angle_tabs[3]:
        st.write("### All Planes Combined")
        fig_all, ax_all = plt.subplots(figsize=(12, 8))
        ax_all.plot(ankle_angles.index, ankle_angles['sagittal_angle'], color='blue', label='Sagittal')
        ax_all.plot(ankle_angles.index, ankle_angles['frontal_angle'], color='green', label='Frontal')
        ax_all.plot(ankle_angles.index, ankle_angles['transverse_angle'], color='purple', label='Transverse')
        ax_all.set_xlabel('Sample')
        ax_all.set_ylabel('Angle (degrees)')
        ax_all.set_title('Ankle Angles in All Planes')
        ax_all.grid(True)
        ax_all.legend()
        st.pyplot(fig_all)
    
    # Calculate ROM metrics
    rom_metrics = rom_analyzer.calculate_rom_metrics(ankle_angles)
    st.subheader("Range of Motion Metrics")
    st.write(rom_metrics)
    
    # Create an instance of the GaitAnalyzer
    gait_analyzer = GaitAnalyzer()

    # Perform gait analysis
    st.subheader('Gait Analysis')
    # Assuming you have shank and foot acceleration data
    shank_acc = imu_data[['imu1_acc_x', 'imu1_acc_y', 'imu1_acc_z']].values
    foot_acc = imu_data[['imu0_acc_x', 'imu0_acc_y', 'imu0_acc_z']].values

    # Detect gait events
    gait_events = gait_analyzer.detect_gait_events(shank_acc, foot_acc)
    st.write('Gait Events:', gait_events)

    # Calculate gait parameters
    gait_parameters = gait_analyzer.calculate_gait_parameters(gait_events, imu_data)
    st.write('Gait Parameters:', gait_parameters)
    
    # Prepare data for prediction
    X, y = load_and_prepare_sequences(imu_data_file)
    X_scaled, y_scaled, scaler_X, scaler_y = standardize_data(X, y)
    
    # Create and load model
    model = create_simple_model(X_scaled.shape[1], X_scaled.shape[2], y_scaled.shape[1])
    # Assume model weights are loaded here
    
    # Make predictions
    predictions = model.predict(X_scaled)
    metrics = evaluate_predictions(y_scaled, predictions, participant_weight)
    
    # Display predictions
    st.subheader('Predictions')
    st.write('Metrics:', metrics)
    
    # Plot results
    st.subheader('Visualizations')
    fig, ax = plt.subplots()
    ax.plot(predictions, label='Predictions')
    ax.plot(y_scaled, label='True Values')
    ax.legend()
    st.pyplot(fig)
    
    # Generate report
    st.subheader('Generate Report')
    if st.button('Download Report'):
        buffer = BytesIO()
        plt.savefig(buffer, format='pdf')
        buffer.seek(0)
        st.download_button('Download Report as PDF', buffer, file_name='report.pdf') 