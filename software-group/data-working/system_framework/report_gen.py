import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Import necessary functions from existing modules
from RoM import calculate_rom
from gait_analysis import perform_gait_analysis
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
    rom_results = calculate_rom(imu_data)
    st.write(rom_results)
    
    # Perform gait analysis
    st.subheader('Gait Analysis')
    gait_results = perform_gait_analysis(imu_data)
    st.write(gait_results)
    
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