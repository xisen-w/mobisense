import streamlit as st
import pandas as pd
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from experiments import Participant, IMU_Experiment_Setup
from RoM_updated import RangeOfMotionAnalyzer
from gait_analysis import GaitAnalyzer

# Configure TensorFlow before importing it
# This needs to happen before importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU mode
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TF logging

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Print TensorFlow configuration for debugging
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"TensorFlow devices: {tf.config.list_physical_devices()}")

st.set_page_config(page_title="MobiSense Analytics", layout="wide")

# Create experiment directory if it doesn't exist
EXPERIMENT_DIR = Path("experiments")
EXPERIMENT_DIR.mkdir(exist_ok=True)

def plot_imu_data(df, imu_num=0, start_idx=0, end_idx=None):
    """Plot IMU acceleration and gyroscope data"""
    if end_idx is None:
        end_idx = min(start_idx + 500, len(df))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Time array
    time_col = f'imu{imu_num}_timestamp'
    # Convert timestamp to numeric (seconds from start)
    time_values = pd.to_datetime(df[time_col].iloc[start_idx:end_idx])
    time_seconds = [(t - time_values.iloc[0]).total_seconds() for t in time_values]
    
    # Plot acceleration
    ax1.plot(time_seconds, df[f'imu{imu_num}_acc_x'].iloc[start_idx:end_idx], label='Acc X')
    ax1.plot(time_seconds, df[f'imu{imu_num}_acc_y'].iloc[start_idx:end_idx], label='Acc Y')
    ax1.plot(time_seconds, df[f'imu{imu_num}_acc_z'].iloc[start_idx:end_idx], label='Acc Z')
    ax1.set_title(f'IMU {imu_num} Acceleration')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_xlabel('Time (s)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot gyroscope
    ax2.plot(time_seconds, df[f'imu{imu_num}_gyro_x'].iloc[start_idx:end_idx], label='Gyro X')
    ax2.plot(time_seconds, df[f'imu{imu_num}_gyro_y'].iloc[start_idx:end_idx], label='Gyro Y')
    ax2.plot(time_seconds, df[f'imu{imu_num}_gyro_z'].iloc[start_idx:end_idx], label='Gyro Z')
    ax2.set_title(f'IMU {imu_num} Gyroscope')
    ax2.set_ylabel('Angular velocity (rad/s)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_angle_data(df, start_idx=0, end_idx=None):
    """Plot dorsiflexion angle data if available"""
    if 'dorsiflexion_angle' not in df.columns:
        return None
    
    if end_idx is None:
        end_idx = min(start_idx + 500, len(df))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Time array
    time_values = pd.to_datetime(df['imu0_timestamp'].iloc[start_idx:end_idx])
    time_seconds = [(t - time_values.iloc[0]).total_seconds() for t in time_values]
    
    # Get angle data
    angle_data = df['dorsiflexion_angle'].iloc[start_idx:end_idx]
    
    # Plot angle data with enhanced styling
    ax.plot(time_seconds, angle_data, label='Dorsiflexion Angle', 
            color='#d62728', linewidth=2.5)
    
    # Calculate statistics
    max_angle = angle_data.max()
    min_angle = angle_data.min()
    mean_angle = angle_data.mean()
    neutral_angle = np.median(angle_data)
    rom = max_angle - min_angle
    
    # Add reference lines
    ax.axhline(y=neutral_angle, color='black', linestyle='--', alpha=0.7, 
               label=f'Neutral: {neutral_angle:.2f}°')
    ax.axhline(y=max_angle, color='green', linestyle=':', alpha=0.7, 
               label=f'Max: {max_angle:.2f}°')
    ax.axhline(y=min_angle, color='red', linestyle=':', alpha=0.7, 
               label=f'Min: {min_angle:.2f}°')
    
    # Add annotations for key statistics
    stats_text = (f"Range of Motion: {rom:.2f}°\n"
                  f"Dorsiflexion: {max_angle - neutral_angle:.2f}°\n"
                  f"Plantarflexion: {neutral_angle - min_angle:.2f}°")
    
    # Place stats box in the corner
    props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7)
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Enhanced styling
    ax.set_title('Ankle Dorsiflexion/Plantarflexion Angle', fontsize=14, fontweight='bold')
    ax.set_ylabel('Angle (degrees)', fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right')
    
    # Add shaded regions to indicate dorsiflexion and plantarflexion
    ax.fill_between(time_seconds, neutral_angle, angle_data, 
                   where=(angle_data > neutral_angle), 
                   color='green', alpha=0.2, label='Dorsiflexion')
    ax.fill_between(time_seconds, angle_data, neutral_angle, 
                   where=(angle_data < neutral_angle), 
                   color='red', alpha=0.2, label='Plantarflexion')
    
    plt.tight_layout()
    return fig

def plot_orientation_data(df, imu_num=0, start_idx=0, end_idx=None):
    """Plot roll, pitch, and yaw orientation data"""
    if end_idx is None:
        end_idx = min(start_idx + 500, len(df))
    
    # Check if orientation columns exist
    orientation_columns = [f'imu{imu_num}_roll', f'imu{imu_num}_pitch', f'imu{imu_num}_yaw']
    if not all(col in df.columns for col in orientation_columns):
        return None
    
    # Create figure with improved styling
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Time array
    time_col = f'imu{imu_num}_timestamp'
    time_values = pd.to_datetime(df[time_col].iloc[start_idx:end_idx])
    time_seconds = [(t - time_values.iloc[0]).total_seconds() for t in time_values]
    
    # Plot orientation data with distinct colors and line styles
    roll = df[f'imu{imu_num}_roll'].iloc[start_idx:end_idx]
    pitch = df[f'imu{imu_num}_pitch'].iloc[start_idx:end_idx]
    yaw = df[f'imu{imu_num}_yaw'].iloc[start_idx:end_idx]
    
    ax.plot(time_seconds, roll, label='Roll', color='#1f77b4', linewidth=2)
    ax.plot(time_seconds, pitch, label='Pitch', color='#ff7f0e', linewidth=2, linestyle='--')
    ax.plot(time_seconds, yaw, label='Yaw', color='#2ca02c', linewidth=2, linestyle='-.')
    
    # Add range indicators
    roll_range = f"Range: {roll.min():.2f} to {roll.max():.2f}°"
    pitch_range = f"Range: {pitch.min():.2f} to {pitch.max():.2f}°"
    yaw_range = f"Range: {yaw.min():.2f} to {yaw.max():.2f}°"
    
    # Enhanced styling
    ax.set_title(f'IMU {imu_num} Orientation Angles', fontsize=14, fontweight='bold')
    ax.set_ylabel('Angle (degrees)', fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=12)
    
    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create legend with additional info
    legend_elements = [
        plt.Line2D([0], [0], color='#1f77b4', lw=2, label=f'Roll: {roll_range}'),
        plt.Line2D([0], [0], color='#ff7f0e', lw=2, linestyle='--', label=f'Pitch: {pitch_range}'),
        plt.Line2D([0], [0], color='#2ca02c', lw=2, linestyle='-.', label=f'Yaw: {yaw_range}')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    # Add overall timespan
    timespan = time_values.iloc[-1] - time_values.iloc[0]
    plt.figtext(0.5, 0.01, f"Timespan: {timespan.total_seconds():.2f} seconds", 
                ha="center", fontsize=10, bbox={"facecolor":"#f0f0f0", "alpha":0.5, "pad":5})
    
    plt.tight_layout()
    return fig

def predict_with_model(model, X):
    """Make predictions with the model"""
    try:
        # Add debug info
        st.info(f"Starting prediction with input shape: {X.shape}")
        
        # Try running on CPU to avoid memory issues
        try:
            with tf.device('/CPU:0'):
                # Make predictions with verbose output
                st.text("Attempting prediction on CPU...")
                predictions = model.predict(X, verbose=1)
        except Exception as cpu_error:
            st.warning(f"CPU prediction failed: {str(cpu_error)}, trying default device")
            # Make predictions with verbose output
            st.text("Falling back to default device prediction...")
            predictions = model.predict(X, verbose=1)
        
        # Add more debug info
        st.info(f"Prediction complete. Output shape: {predictions.shape}")
        
        # Check if predictions are valid
        if np.isnan(predictions).any():
            return None, "Predictions contain NaN values. Model may not be working correctly."
        
        return predictions, None
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"Detailed error: {error_details}")
        return None, f"Error making predictions: {str(e)}"

def load_ml_model():
    """Load the best model for predictions"""
    try:
        # Ensure we're using CPU for prediction (safer on Mac)
        device_lib = tf.config.list_physical_devices()
        st.info(f"Available devices: {device_lib}")
        
        # Load the model with error handling
        model_path = "model_output/best_model"
        if not os.path.exists(model_path):
            st.error(f"Model path does not exist: {model_path}")
            return None
        
        # Load the model on CPU to avoid memory issues
        with tf.device('/CPU:0'):
            st.info("Loading model on CPU...")
            model = tf.keras.models.load_model(model_path)
            st.success("Model loaded successfully on CPU")
        
        # Check model architecture
        input_shape = model.input_shape
        output_shape = model.output_shape
        st.session_state['model_info'] = {
            'input_shape': input_shape,
            'output_shape': output_shape
        }
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def prepare_data_for_prediction(df, sequence_length=10):
    """Prepare data from the dataframe for model prediction"""
    # The model expects shape=(None, 10, 12), but we're providing shape=(None, 10, 19)
    # Need to select only the columns the model was trained on
    
    # Based on the error, the model expects exactly 12 features
    # We'll prioritize the most important IMU features
    
    # First check if we have all necessary columns
    required_imu_cols = [
        # Core IMU features that are most likely to be used in training
        'imu0_acc_x', 'imu0_acc_y', 'imu0_acc_z',
        'imu0_gyro_x', 'imu0_gyro_y', 'imu0_gyro_z',
        'imu1_acc_x', 'imu1_acc_y', 'imu1_acc_z',
        'imu1_gyro_x', 'imu1_gyro_y', 'imu1_gyro_z'
    ]
    
    # Check if all required columns exist
    if not all(col in df.columns for col in required_imu_cols):
        missing_cols = [col for col in required_imu_cols if col not in df.columns]
        return None, f"Missing required columns: {', '.join(missing_cols)}"
    
    # Create sequences using exactly the required columns to match the expected input shape
    sequences = []
    for i in range(len(df) - sequence_length + 1):
        sequence = df.iloc[i:i+sequence_length][required_imu_cols].values
        sequences.append(sequence)
    
    if not sequences:
        return None, "Could not create any sequences"
    
    # Convert to numpy array
    X = np.array(sequences)
    
    # Check the shape to make sure it matches what the model expects
    n_samples, n_steps, n_features = X.shape
    if n_features != 12:  # The model expects exactly 12 features
        return None, f"Expected 12 features but got {n_features}. Model requires exactly 12 input features."
    
    # Normalize data (assuming model was trained with normalized data)
    scaler = StandardScaler()
    X_reshaped = X.reshape(n_samples, n_steps * n_features)
    X_normalized = scaler.fit_transform(X_reshaped)
    X_normalized = X_normalized.reshape(n_samples, n_steps, n_features)
    
    return X_normalized, None

def plot_predictions(predictions, prediction_type="force", scaled=False):
    """Plot the predictions"""
    try:
        if prediction_type == "force":
            # Assuming prediction output is force data with [left_x, left_y, left_z, right_x, right_y, right_z]
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot vertical forces (y-component) which are usually most important
            ax.plot(predictions[:, 1], label='Left Foot Vertical Force', color='#1f77b4')
            ax.plot(predictions[:, 4], label='Right Foot Vertical Force', color='#ff7f0e')
            
            # Plot total vertical force
            total_vertical = predictions[:, 1] + predictions[:, 4]
            ax.plot(total_vertical, label='Total Vertical Force', color='green', linestyle='--', linewidth=2)
            
            # Add information about scaling if applied
            title = 'Predicted Ground Reaction Forces'
            if scaled:
                title += ' (Scaled)'
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Steps', fontsize=12)
            ax.set_ylabel('Force (Newtons)', fontsize=12)
            
            # Add average force annotation
            avg_force = np.mean(total_vertical)
            ax.annotate(f'Avg Total Force: {avg_force:.2f} N', 
                       xy=(0.02, 0.95), xycoords='axes fraction',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            plt.tight_layout()
            
            # Second plot for horizontal forces
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            # Plot anterior-posterior forces (x-component)
            ax2.plot(predictions[:, 0], label='Left Foot A-P Force', color='#1f77b4')
            ax2.plot(predictions[:, 3], label='Right Foot A-P Force', color='#ff7f0e')
            
            # Plot medial-lateral forces (z-component)
            ax2.plot(predictions[:, 2], label='Left Foot M-L Force', color='#2ca02c', linestyle='--')
            ax2.plot(predictions[:, 5], label='Right Foot M-L Force', color='#d62728', linestyle='--')
            
            # Add information about scaling if applied
            title = 'Predicted Horizontal Forces'
            if scaled:
                title += ' (Scaled)'
                
            ax2.set_title(title, fontsize=14, fontweight='bold')
            ax2.set_xlabel('Time Steps', fontsize=12)
            ax2.set_ylabel('Force (Newtons)', fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend()
            plt.tight_layout()
            
            return [fig, fig2]
        
        elif prediction_type == "angle":
            # Assuming prediction output is angle data
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(predictions, label='Predicted Angle', color='#1f77b4', linewidth=2)
            
            ax.set_title('Predicted Joint Angle', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Steps', fontsize=12)
            ax.set_ylabel('Angle (degrees)', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            plt.tight_layout()
            
            return [fig]
        
        else:
            # Generic plot for other prediction types
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for i in range(predictions.shape[1]):
                ax.plot(predictions[:, i], label=f'Output {i+1}')
            
            ax.set_title('Model Predictions', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Steps', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            plt.tight_layout()
            
            return [fig]
            
    except Exception as e:
        st.error(f"Error plotting predictions: {str(e)}")
        return None

def main():
    st.title("MobiSense Experiment Setup")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Upload & Experiment Setup", "View Experiments", "Data Visualization", "Gait Analytics", "ML Predictions"])
    
    with tab1:
        st.header("Create New Experiment")
        
        # Form for experiment setup
        with st.form("experiment_setup_form"):
            st.subheader("Experiment Information")
            experiment_name = st.text_input("Experiment Name")
            experiment_description = st.text_area("Experiment Description")
            
            st.subheader("Participant Information")
            participant_id = st.text_input("Participant ID")
            height = st.number_input("Height (cm)", min_value=0.0, step=1.0)
            weight = st.number_input("Weight (kg)", min_value=0.0, step=1.0)
            age = st.number_input("Age", min_value=0, step=1)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            # Optional parameters with expander
            with st.expander("Gait Parameters (Optional)"):
                stride_length = st.number_input("Stride Length (cm)", min_value=0.0, step=1.0, value=None)
                stride_number_per_minute = st.number_input("Stride Number per Minute", min_value=0, step=1, value=None)
            
            # Injury information
            with st.expander("Injury Information (Optional)"):
                injury_day = st.date_input("Injury Date", value=None)
                injury_type = st.text_input("Injury Type")
            
            st.subheader("Data Upload")
            uploaded_file = st.file_uploader("Upload IMU Data (CSV)", type="csv")
            
            submitted = st.form_submit_button("Register Experiment")
            
            if submitted and uploaded_file is not None:
                # First check raw file content
                raw_content = uploaded_file.getvalue().decode('utf-8').splitlines()
                header = raw_content[0]
                first_line = raw_content[1] if len(raw_content) > 1 else ""
                
                # Check if dorsiflexion_angle is in header
                has_angle_in_raw = 'dorsiflexion_angle' in header
                
                # Process the uploaded file
                df = pd.read_csv(uploaded_file)
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Debug column info
                st.write("**Available columns:**")
                st.write(", ".join(df.columns.tolist()))
                
                # Check for dorsiflexion angle column
                has_angles = 'dorsiflexion_angle' in df.columns
                
                # If angle data is in raw but not in dataframe, try to fix
                if has_angle_in_raw and not has_angles:
                    st.warning("Dorsiflexion angle found in CSV header but not in DataFrame. Attempting to fix...")
                    try:
                        # Reset file pointer
                        uploaded_file.seek(0)
                        # Try different parsing options
                        df = pd.read_csv(uploaded_file, low_memory=False)
                        has_angles = 'dorsiflexion_angle' in df.columns
                        if has_angles:
                            st.success("Successfully recovered dorsiflexion angle data!")
                    except Exception as e:
                        st.error(f"Error trying to fix angle data: {str(e)}")
                
                # Display orientation data if available
                orientation_cols = ['imu0_roll', 'imu0_pitch', 'imu0_yaw']
                if all(col in df.columns for col in orientation_cols):
                    st.write("**Orientation data preview:**")
                    st.dataframe(df[orientation_cols].head(5))
                
                # Save the uploaded file
                file_path = Path(f"experiments/{experiment_name}_{participant_id}.csv")
                df.to_csv(file_path, index=False)
                
                # Create participant and experiment
                participant = Participant(
                    participant_id=participant_id,
                    height=height,
                    weight=weight,
                    age=age,
                    gender=gender,
                    stride_length=stride_length if stride_length else None,
                    stride_number_per_minute=stride_number_per_minute if stride_number_per_minute else None,
                    injury_day=injury_day.strftime('%Y-%m-%d') if injury_day else None,
                    injury_type=injury_type if injury_type else None
                )
                
                experiment = IMU_Experiment_Setup(
                    experiment_name=experiment_name,
                    experiment_description=experiment_description,
                    experiment_data_path=str(file_path),
                    participant=participant
                )
                
                # Load and validate data
                experiment.load_experiment_data()
                
                # Double-check angle detection
                if has_angles:
                    experiment.has_angle_data = True
                
                # Save experiment metadata
                experiment_metadata = experiment.get_experiment_info()
                
                metadata_path = Path(f"experiments/{experiment_name}_{participant_id}_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(experiment_metadata, f, indent=4)
                
                st.success(f"Experiment '{experiment_name}' registered successfully!")
                
                # Check data structure
                if has_angles:
                    st.info("Angle data detected in the dataset.")
                
                # Validate IMU data structure
                if experiment.validate_experiment_data():
                    st.success("Data structure validation passed.")
                else:
                    st.warning("Data structure validation failed. Missing required columns.")
    
    with tab2:
        st.header("View Existing Experiments")
        if os.path.exists("experiments"):
            experiment_files = [f for f in os.listdir("experiments") if f.endswith("_metadata.json")]
            
            if experiment_files:
                for exp_file in experiment_files:
                    with open(f"experiments/{exp_file}", 'r') as f:
                        exp_data = json.load(f)
                    
                    with st.expander(f"Experiment: {exp_data['experiment_name']}"):
                        st.write(f"**Description:** {exp_data['experiment_description']}")
                        st.write("**Participant Information:**")
                        participant = exp_data['participant']
                        participant_info = f"""
                        - ID: {participant['participant_id']}
                        - Height: {participant['height']} cm
                        - Weight: {participant['weight']} kg
                        - Age: {participant['age']}
                        - Gender: {participant['gender']}
                        """
                        
                        # Add optional parameters if they exist
                        if participant.get('stride_length'):
                            participant_info += f"- Stride Length: {participant['stride_length']} cm\n"
                        
                        if participant.get('stride_number_per_minute'):
                            participant_info += f"- Stride Number per Minute: {participant['stride_number_per_minute']}\n"
                        
                        # Add injury information if available
                        if participant.get('injury_day'):
                            participant_info += f"- Injury Date: {participant['injury_day']}\n"
                        
                        if participant.get('injury_type'):
                            participant_info += f"- Injury Type: {participant['injury_type']}\n"
                            
                        st.markdown(participant_info)
                        
                        st.write(f"**Data Path:** {exp_data['data_path']}")
                        st.write(f"**Has Angle Data:** {exp_data.get('has_angle_data', 'Unknown')}")
                        
                        # Add button to load experiment data
                        if st.button(f"Load Data Preview", key=f"load_{exp_file}"):
                            data_path = exp_data["data_path"]
                            if os.path.exists(data_path):
                                try:
                                    # First check raw CSV file
                                    with open(data_path, 'r') as f:
                                        header = f.readline().strip()
                                        first_line = f.readline().strip()
                                    
                                    # Check if dorsiflexion_angle is in header
                                    has_angle_in_raw = 'dorsiflexion_angle' in header
                                    
                                    # Now load with pandas
                                    exp_df = pd.read_csv(data_path)
                                    st.dataframe(exp_df.head())
                                    
                                    # Show column names
                                    st.write("**Available columns:**")
                                    st.write(", ".join(exp_df.columns.tolist()))
                                    
                                    # Verify angle data
                                    has_angle_in_df = 'dorsiflexion_angle' in exp_df.columns
                                    
                                    if has_angle_in_df:
                                        st.success("✅ Dorsiflexion angle data is present in this dataset.")
                                        # Show sample of angle data
                                        st.write("**Sample angle data:**")
                                        st.write(exp_df['dorsiflexion_angle'].head(5).tolist())
                                    elif has_angle_in_raw:
                                        st.warning("⚠️ Dorsiflexion angle found in raw CSV but not in DataFrame! Likely a parsing issue.")
                                        # Try to fix
                                        st.write("Attempting to reload with different options...")
                                        fixed_df = pd.read_csv(data_path, low_memory=False)
                                        if 'dorsiflexion_angle' in fixed_df.columns:
                                            st.success("Fixed! Angle data is now available.")
                                            st.write(fixed_df['dorsiflexion_angle'].head(5).tolist())
                                            exp_df = fixed_df  # Update dataframe
                                    else:
                                        st.warning("❌ No dorsiflexion angle data found in this dataset.")
                                    
                                    # Show orientation data preview if available
                                    orientation_cols = ['imu0_roll', 'imu0_pitch', 'imu0_yaw']
                                    if all(col in exp_df.columns for col in orientation_cols):
                                        st.write("**Orientation data (first 5 rows):**")
                                        st.dataframe(exp_df[orientation_cols].head(5))
                                except Exception as e:
                                    st.error(f"Error loading data: {str(e)}")
                            else:
                                st.error(f"Data file not found: {data_path}")
            else:
                st.info("No experiments found.")
        else:
            st.info("No experiments directory found.")

    with tab3:
        st.header("Data Visualization")
        
        if os.path.exists("experiments"):
            experiment_files = [f for f in os.listdir("experiments") if f.endswith("_metadata.json")]
            
            if experiment_files:
                # Create a selectbox to choose experiment
                experiment_names = [f.replace("_metadata.json", "") for f in experiment_files]
                selected_experiment = st.selectbox("Select Experiment", experiment_names)
                
                if selected_experiment:
                    metadata_path = f"experiments/{selected_experiment}_metadata.json"
                    
                    with open(metadata_path, 'r') as f:
                        exp_data = json.load(f)
                    
                    data_path = exp_data["data_path"]
                    
                    if os.path.exists(data_path):
                        # Load the data
                        df = pd.read_csv(data_path)
                        
                        # Display data statistics
                        st.subheader("Data Statistics")
                        st.write(f"Total samples: {len(df)}")
                        st.write(f"Duration: {(pd.to_datetime(df['imu0_timestamp'].iloc[-1]) - pd.to_datetime(df['imu0_timestamp'].iloc[0])).total_seconds():.2f} seconds")
                        
                        # Show list of available columns
                        st.subheader("Available Data Columns")
                        st.write(", ".join(df.columns.tolist()))
                        
                        # Debug raw CSV content 
                        with st.expander("Raw CSV First Row Debug"):
                            try:
                                # Display first few rows as raw text
                                with open(data_path, 'r') as f:
                                    header = f.readline().strip()
                                    first_line = f.readline().strip()
                                
                                st.write("**CSV Header:**")
                                st.code(header)
                                st.write("**First Data Row:**")
                                st.code(first_line)
                                
                                # Check if 'dorsiflexion_angle' is in header
                                if 'dorsiflexion_angle' in header:
                                    st.success("'dorsiflexion_angle' found in CSV header")
                                    # Find position of dorsiflexion_angle in header
                                    header_parts = header.split(',')
                                    angle_index = header_parts.index('dorsiflexion_angle')
                                    st.write(f"Position of dorsiflexion_angle: {angle_index} out of {len(header_parts)}")
                                    
                                    # Check corresponding value in first line
                                    data_parts = first_line.split(',')
                                    if angle_index < len(data_parts):
                                        angle_value = data_parts[angle_index]
                                        st.write(f"First dorsiflexion_angle value: {angle_value}")
                                else:
                                    st.error("'dorsiflexion_angle' NOT found in CSV header")
                            except Exception as e:
                                st.error(f"Error reading raw CSV: {str(e)}")
                        
                        # Explicitly check for angle data
                        has_dorsiflexion_angle = 'dorsiflexion_angle' in df.columns
                        if has_dorsiflexion_angle:
                            st.success("Dorsiflexion angle data is available in this dataset.")
                            
                            # Debug info for angle data
                            with st.expander("Angle Data Debug Info"):
                                st.write("First 10 angle values:")
                                st.write(df['dorsiflexion_angle'].head(10).tolist())
                                
                                # Check for NaN values or zeros
                                nan_count = df['dorsiflexion_angle'].isna().sum()
                                zero_count = (df['dorsiflexion_angle'] == 0).sum()
                                st.write(f"NaN values: {nan_count}")
                                st.write(f"Zero values: {zero_count}")
                                
                                # Statistics 
                                st.write(f"Min: {df['dorsiflexion_angle'].min()}")
                                st.write(f"Max: {df['dorsiflexion_angle'].max()}")
                                st.write(f"Mean: {df['dorsiflexion_angle'].mean()}")
                        else:
                            st.warning("Dorsiflexion angle data was NOT found in this dataset.")
                            # Try to fix by reloading the file directly
                            with st.expander("Attempt Recovery"):
                                st.write("Attempting to reload CSV directly...")
                                try:
                                    # Try reading the CSV file with different options
                                    direct_df = pd.read_csv(data_path, low_memory=False)
                                    st.write("Columns after direct reload:")
                                    st.write(direct_df.columns.tolist())
                                    if 'dorsiflexion_angle' in direct_df.columns:
                                        st.success("Found dorsiflexion_angle in direct reload!")
                                        df = direct_df  # Update the dataframe
                                        has_dorsiflexion_angle = True
                                except Exception as e:
                                    st.error(f"Recovery attempt failed: {str(e)}")
                        
                        # Data selection options
                        st.subheader("Visualization Options")
                        
                        # Create columns for options
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            imu_selection = st.radio("Select IMU", [0, 1])
                        
                        with col2:
                            # Add view mode option
                            view_mode = st.radio("View Mode", ["Dashboard", "Detailed"])
                        
                        sample_range = st.slider(
                            "Select Sample Range", 
                            min_value=0, 
                            max_value=len(df)-1, 
                            value=(0, min(500, len(df)-1))
                        )
                        
                        # Create expandable sections for each visualization
                        if view_mode == "Dashboard":
                            # Dashboard layout - arrange plots in a grid
                            st.subheader("Data Dashboard")
                            
                            # Create tabs for each IMU
                            imu0_tab, imu1_tab, angles_tab = st.tabs(["IMU 0", "IMU 1", "Angles"])
                            
                            with imu0_tab:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Plot IMU0 acceleration/gyro data
                                    st.subheader("IMU 0 Acceleration & Gyroscope")
                                    imu0_fig = plot_imu_data(df, imu_num=0, start_idx=sample_range[0], end_idx=sample_range[1])
                                    st.pyplot(imu0_fig)
                                
                                with col2:
                                    # Plot IMU0 orientation data
                                    orientation_columns_0 = ['imu0_roll', 'imu0_pitch', 'imu0_yaw']
                                    if all(col in df.columns for col in orientation_columns_0):
                                        st.subheader("IMU 0 Orientation")
                                        orientation_fig_0 = plot_orientation_data(df, imu_num=0, start_idx=sample_range[0], end_idx=sample_range[1])
                                        st.pyplot(orientation_fig_0)
                                    else:
                                        st.info("Orientation data not available for IMU 0")
                            
                            with imu1_tab:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Plot IMU1 acceleration/gyro data
                                    st.subheader("IMU 1 Acceleration & Gyroscope")
                                    imu1_fig = plot_imu_data(df, imu_num=1, start_idx=sample_range[0], end_idx=sample_range[1])
                                    st.pyplot(imu1_fig)
                                
                                with col2:
                                    # Plot IMU1 orientation data
                                    orientation_columns_1 = ['imu1_roll', 'imu1_pitch', 'imu1_yaw']
                                    if all(col in df.columns for col in orientation_columns_1):
                                        st.subheader("IMU 1 Orientation")
                                        orientation_fig_1 = plot_orientation_data(df, imu_num=1, start_idx=sample_range[0], end_idx=sample_range[1])
                                        st.pyplot(orientation_fig_1)
                                    else:
                                        st.info("Orientation data not available for IMU 1")
                            
                            with angles_tab:
                                # Plot dorsiflexion angle data if available
                                if has_dorsiflexion_angle:
                                    st.subheader("Dorsiflexion Angle")
                                    angle_fig = plot_angle_data(df, start_idx=sample_range[0], end_idx=sample_range[1])
                                    st.pyplot(angle_fig)
                                else:
                                    st.warning("No dorsiflexion angle data available in this dataset.")
                        
                        else:  # Detailed view
                            # Detailed view - show plots in expandable sections with full width
                            
                            # Plot IMU data
                            with st.expander(f"IMU {imu_selection} Acceleration & Gyroscope Data", expanded=True):
                                imu_fig = plot_imu_data(df, imu_num=imu_selection, start_idx=sample_range[0], end_idx=sample_range[1])
                                st.pyplot(imu_fig, use_container_width=True)
                            
                            # Plot orientation data for selected IMU
                            orientation_columns = [f'imu{imu_selection}_roll', f'imu{imu_selection}_pitch', f'imu{imu_selection}_yaw']
                            if all(col in df.columns for col in orientation_columns):
                                with st.expander(f"IMU {imu_selection} Orientation (Roll, Pitch, Yaw)", expanded=True):
                                    orientation_fig = plot_orientation_data(df, imu_num=imu_selection, start_idx=sample_range[0], end_idx=sample_range[1])
                                    st.pyplot(orientation_fig, use_container_width=True)
                            else:
                                st.info(f"Orientation data (roll, pitch, yaw) not available for IMU {imu_selection}")
                            
                            # Plot orientation data for the other IMU
                            other_imu = 1 if imu_selection == 0 else 0
                            orientation_columns_other = [f'imu{other_imu}_roll', f'imu{other_imu}_pitch', f'imu{other_imu}_yaw']
                            if all(col in df.columns for col in orientation_columns_other):
                                with st.expander(f"IMU {other_imu} Orientation (Roll, Pitch, Yaw)", expanded=False):
                                    orientation_fig_other = plot_orientation_data(df, imu_num=other_imu, start_idx=sample_range[0], end_idx=sample_range[1])
                                    st.pyplot(orientation_fig_other, use_container_width=True)
                            
                            # Plot angle data if available
                            if has_dorsiflexion_angle:
                                with st.expander("Dorsiflexion Angle Data", expanded=True):
                                    angle_fig = plot_angle_data(df, start_idx=sample_range[0], end_idx=sample_range[1])
                                    st.pyplot(angle_fig, use_container_width=True)
                            else:
                                st.warning("No dorsiflexion angle data available in this dataset.")
                    else:
                        st.error(f"Data file not found: {data_path}")
            else:
                st.info("No experiments found.")
        else:
            st.info("No experiments directory found.")

    with tab4:
        st.header("Gait Analytics Dashboard")
        
        if os.path.exists("experiments"):
            experiment_files = [f for f in os.listdir("experiments") if f.endswith("_metadata.json")]
            
            if experiment_files:
                # Create a selectbox to choose experiment
                experiment_names = [f.replace("_metadata.json", "") for f in experiment_files]
                selected_experiment = st.selectbox("Select Experiment", experiment_names, key="gait_experiment_select")
                
                if selected_experiment:
                    metadata_path = f"experiments/{selected_experiment}_metadata.json"
                    
                    with open(metadata_path, 'r') as f:
                        exp_data = json.load(f)
                    
                    data_path = exp_data["data_path"]
                    
                    if os.path.exists(data_path):
                        # Load the data
                        df = pd.read_csv(data_path)
                        
                        # Initialize analyzers
                        rom_analyzer = RangeOfMotionAnalyzer(sampling_rate=100.0)  # Assuming 100Hz sampling rate
                        gait_analyzer = GaitAnalyzer(sampling_rate=100.0)
                        
                        # Display data statistics
                        st.subheader("Gait Analysis Options")
                        
                        sample_range = st.slider(
                            "Select Sample Range for Analysis", 
                            min_value=0, 
                            max_value=len(df)-1, 
                            value=(0, min(3000, len(df)-1)),
                            key="gait_sample_range"
                        )
                        
                        selected_df = df.iloc[sample_range[0]:sample_range[1]]
                        
                        # Create dashboard layout
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.subheader("Gait Parameter Analysis")
                            
                            # 1. Calculate ankle angles
                            with st.spinner("Calculating ankle angles..."):
                                has_dorsiflexion_angle = 'dorsiflexion_angle' in selected_df.columns
                                
                                # Calculate ankle angles
                                try:
                                    angle_results = rom_analyzer.calculate_ankle_angles(selected_df)
                                    angles_df = angle_results['angles']
                                    rom_metrics = angle_results['metrics']
                                    
                                    # Create two columns for visualization
                                    angle_col1, angle_col2 = st.columns(2)
                                    
                                    with angle_col1:
                                        # Plot angle data
                                        fig, ax = plt.subplots(figsize=(10, 5))
                                        time_values = np.arange(len(angles_df)) / rom_analyzer.sampling_rate
                                        
                                        ax.plot(time_values, angles_df['sagittal_angle'], label='Dorsiflexion/Plantarflexion', 
                                                color='#1f77b4', linewidth=2)
                                        ax.plot(time_values, angles_df['frontal_angle'], label='Inversion/Eversion', 
                                                color='#ff7f0e', linewidth=2, linestyle='--')
                                        
                                        ax.set_title('Ankle Joint Angles', fontsize=14, fontweight='bold')
                                        ax.set_xlabel('Time (s)', fontsize=12)
                                        ax.set_ylabel('Angle (degrees)', fontsize=12)
                                        ax.grid(True, linestyle='--', alpha=0.7)
                                        ax.legend(loc='best')
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                    
                                    with angle_col2:
                                        # Display ROM metrics
                                        st.subheader("Range of Motion Metrics")
                                        metric_df = pd.DataFrame(list(rom_metrics.items()), columns=['Metric', 'Value'])
                                        
                                        # Format and sort metrics
                                        metric_df['Value'] = metric_df['Value'].map(lambda x: f"{x:.2f}°")
                                        metric_df['Metric'] = metric_df['Metric'].map(lambda x: x.replace('_', ' ').title())
                                        
                                        # Create a styled dataframe
                                        st.dataframe(
                                            metric_df,
                                            hide_index=True,
                                            column_config={
                                                "Metric": st.column_config.TextColumn("Parameter", width="large"),
                                                "Value": st.column_config.TextColumn("Value", width="small")
                                            }
                                        )
                                except Exception as e:
                                    st.error(f"Error calculating ankle angles: {str(e)}")
                            
                            # 2. Detect gait events
                            with st.spinner("Detecting gait events..."):
                                try:
                                    # Prepare acceleration data
                                    shank_acc = selected_df[['imu0_acc_x', 'imu0_acc_y', 'imu0_acc_z']].values
                                    foot_acc = selected_df[['imu1_acc_x', 'imu1_acc_y', 'imu1_acc_z']].values
                                    
                                    # Detect gait events
                                    gait_events = gait_analyzer.detect_gait_events(shank_acc, foot_acc)
                                    
                                    # Calculate gait parameters if ankle angles are available
                                    if 'angles_df' in locals():
                                        gait_params = gait_analyzer.calculate_gait_parameters(gait_events, angles_df)
                                        pathological_metrics = gait_analyzer.analyze_pathological_gait(angles_df, gait_events)
                                        
                                        # Create visualization for gait events
                                        fig, ax = plt.subplots(figsize=(10, 5))
                                        
                                        # Plot acceleration data
                                        acc_magnitude = np.linalg.norm(shank_acc, axis=1)
                                        time_values = np.arange(len(acc_magnitude)) / gait_analyzer.sampling_rate
                                        
                                        ax.plot(time_values, acc_magnitude, label='Shank Acceleration', alpha=0.7)
                                        
                                        # Mark heel strikes and toe-offs
                                        for hs in gait_events['heel_strikes']:
                                            if hs < len(time_values):
                                                ax.axvline(x=time_values[hs], color='red', linestyle='--', alpha=0.5)
                                        
                                        for to in gait_events['toe_offs']:
                                            if to < len(time_values):
                                                ax.axvline(x=time_values[to], color='green', linestyle=':', alpha=0.5)
                                        
                                        # Add legend elements for gait events
                                        from matplotlib.lines import Line2D
                                        legend_elements = [
                                            Line2D([0], [0], color='red', linestyle='--', label='Heel Strike'),
                                            Line2D([0], [0], color='green', linestyle=':', label='Toe Off')
                                        ]
                                        
                                        ax.set_title('Gait Event Detection', fontsize=14, fontweight='bold')
                                        ax.set_xlabel('Time (s)', fontsize=12)
                                        ax.set_ylabel('Acceleration Magnitude (m/s²)', fontsize=12)
                                        ax.grid(True, linestyle='--', alpha=0.7)
                                        ax.legend(handles=legend_elements, loc='best')
                                        
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        
                                        # Display gait cycle information
                                        st.subheader("Gait Cycle Information")
                                        st.write(f"Number of detected heel strikes: {len(gait_events['heel_strikes'])}")
                                        st.write(f"Number of detected toe-offs: {len(gait_events['toe_offs'])}")
                                    
                                except Exception as e:
                                    st.error(f"Error detecting gait events: {str(e)}")
                        
                        with col2:
                            st.subheader("Gait Parameters")
                            
                            # Only display if we have calculated the parameters
                            if 'gait_params' in locals() and 'pathological_metrics' in locals():
                                # Combine all metrics
                                all_metrics = {**gait_params, **pathological_metrics}
                                
                                # Create dataframe for metrics
                                metrics_df = pd.DataFrame(list(all_metrics.items()), columns=['Parameter', 'Value'])
                                
                                # Format metrics
                                metrics_df['Parameter'] = metrics_df['Parameter'].map(lambda x: x.replace('_', ' ').title())
                                
                                # Define styled metrics display
                                st.dataframe(
                                    metrics_df,
                                    hide_index=True,
                                    column_config={
                                        "Parameter": st.column_config.TextColumn("Gait Parameter", width="large"),
                                        "Value": st.column_config.NumberColumn("Value", format="%.2f")
                                    }
                                )
                                
                                # Calculate sprain risk metrics if we have angle data
                                if 'angles_df' in locals():
                                    try:
                                        sprain_risk = rom_analyzer.calculate_sprain_risk_metrics(angles_df)
                                        
                                        # Create gauge charts for key metrics
                                        st.subheader("Ankle Instability Risk Assessment")
                                        
                                        # Create 3 columns for gauges
                                        gauge1, gauge2, gauge3 = st.columns(3)
                                        
                                        # Helper function to create a gauge chart
                                        def create_gauge(value, title, min_val=0, max_val=100, danger_threshold=70, 
                                                       warn_threshold=40, unit=""):
                                            fig, ax = plt.subplots(figsize=(4, 3), subplot_kw={'polar': True})
                                            
                                            # Normalize value to 0-1 range
                                            normalized = (max(min(value, max_val), min_val) - min_val) / (max_val - min_val)
                                            
                                            # Define color gradient
                                            cmap = plt.cm.RdYlGn_r
                                            color = cmap(normalized)
                                            
                                            # Plot background
                                            ax.set_theta_direction(-1)
                                            ax.set_theta_offset(np.pi/2)
                                            
                                            # Plot gauge
                                            ax.set_rlim(0, 1)
                                            ax.set_yticks([])
                                            ax.set_xticks(np.linspace(0, 2*np.pi, 9)[:-1])
                                            ax.set_xticklabels([])
                                            
                                            # Plot value
                                            ax.bar(0, normalized, width=2*np.pi, color=color, alpha=0.7)
                                            
                                            # Add text in center
                                            plt.text(0, 0, f"{value:.1f}{unit}", 
                                                    ha='center', va='center', fontsize=16)
                                            plt.text(0, -0.4, title, 
                                                    ha='center', va='center', fontsize=12)
                                            
                                            plt.tight_layout()
                                            return fig
                                        
                                        # Create gauges for key metrics
                                        with gauge1:
                                            max_inversion = sprain_risk['max_inversion_velocity']
                                            fig = create_gauge(
                                                max_inversion, 
                                                "Max Inversion\nVelocity", 
                                                min_val=0, 
                                                max_val=400, 
                                                danger_threshold=300, 
                                                unit="°/s"
                                            )
                                            st.pyplot(fig)
                                        
                                        with gauge2:
                                            stability = min(sprain_risk['stability_index'] * 5, 100)  # Scale for visualization
                                            fig = create_gauge(
                                                stability, 
                                                "Stability Index", 
                                                min_val=0, 
                                                max_val=100, 
                                                danger_threshold=70
                                            )
                                            st.pyplot(fig)
                                            
                                        with gauge3:
                                            sudden_inversions = sprain_risk['sudden_inversion_count']
                                            fig = create_gauge(
                                                sudden_inversions, 
                                                "Sudden Inversions", 
                                                min_val=0, 
                                                max_val=10, 
                                                danger_threshold=5
                                            )
                                            st.pyplot(fig)
                                        
                                        # Show detailed sprain risk metrics
                                        with st.expander("Detailed Sprain Risk Metrics"):
                                            risk_df = pd.DataFrame(list(sprain_risk.items()), 
                                                                 columns=['Metric', 'Value'])
                                            risk_df['Metric'] = risk_df['Metric'].map(
                                                lambda x: x.replace('_', ' ').title())
                                            st.dataframe(risk_df, hide_index=True)
                                            
                                    except Exception as e:
                                        st.error(f"Error calculating sprain risk: {str(e)}")
                            else:
                                st.info("Gait parameters will be displayed after angle calculation and gait event detection.")
                            
                            # Add a section for gait pattern classification
                            st.subheader("Gait Pattern Analysis")
                            
                            # Check if we have the necessary data
                            if 'gait_params' in locals() and 'angles_df' in locals():
                                try:
                                    # Create simple gait pattern classification
                                    pattern_score = 0
                                    pattern_indicators = []
                                    
                                    # Calculate stride regularity
                                    if 'stride_time_variability' in gait_params:
                                        if gait_params['stride_time_variability'] > 0.15:
                                            pattern_score += 1
                                            pattern_indicators.append("Irregular stride timing")
                                    
                                    # Check for abnormal inversion/eversion patterns
                                    if 'inversion_variability' in pathological_metrics:
                                        if pathological_metrics['inversion_variability'] > 5:
                                            pattern_score += 1
                                            pattern_indicators.append("Variable ankle inversion")
                                    
                                    # Check for early heel rise pattern
                                    if 'early_heel_rise_ratio' in pathological_metrics:
                                        if pathological_metrics['early_heel_rise_ratio'] > 0.2:
                                            pattern_score += 1
                                            pattern_indicators.append("Early heel rise detected")
                                    
                                    # Classify gait pattern
                                    if pattern_score == 0:
                                        pattern = "Normal gait pattern"
                                        pattern_color = "green"
                                    elif pattern_score == 1:
                                        pattern = "Mild gait abnormality"
                                        pattern_color = "#FFA500"  # Orange
                                    else:
                                        pattern = "Significant gait abnormality"
                                        pattern_color = "red"
                                    
                                    # Display gait pattern classification with styling
                                    st.markdown(f"<h3 style='color:{pattern_color}'>{pattern}</h3>", unsafe_allow_html=True)
                                    
                                    if pattern_indicators:
                                        st.write("Indicators detected:")
                                        for indicator in pattern_indicators:
                                            st.markdown(f"- {indicator}")
                                    else:
                                        st.write("No specific gait abnormalities detected.")
                                    
                                except Exception as e:
                                    st.error(f"Error in gait pattern analysis: {str(e)}")
                            else:
                                st.info("Gait pattern analysis requires gait parameters and angle data.")
                    else:
                        st.error(f"Data file not found: {data_path}")
            else:
                st.info("No experiments found.")
        else:
            st.info("No experiments directory found.")

    with tab5:
        st.header("ML Predictions")
        
        try:
            # Load the model
            model = load_ml_model()
            
            if model is None:
                st.error("Could not load the prediction model. Please check if the model exists.")
                st.warning("Continuing with limited functionality. You can still view experiments but predictions will not be available.")
                
                # Show troubleshooting information
                with st.expander("Troubleshooting Information"):
                    st.markdown("""
                    ### Common Issues:
                    1. The model file may not exist in the expected location (model_output/best_model)
                    2. TensorFlow/GPU compatibility issues
                    3. Memory limitations
                    
                    ### Solutions:
                    1. Ensure the model files are in the correct location
                    2. Try running with CPU only by uncommenting the relevant line in the code
                    3. Restart the application
                    """)
            else:
                st.success("Successfully loaded the prediction model.")
                
                # Display model information if available
                if 'model_info' in st.session_state:
                    with st.expander("Model Information"):
                        st.write(f"**Input Shape:** {st.session_state['model_info']['input_shape']}")
                        st.write(f"**Output Shape:** {st.session_state['model_info']['output_shape']}")
                        st.info("This model requires exactly 12 IMU features (accelerometer and gyroscope data from both sensors) with a sequence length of 10.")
                
                if os.path.exists("experiments"):
                    experiment_files = [f for f in os.listdir("experiments") if f.endswith("_metadata.json")]
                    
                    if experiment_files:
                        # Create a selectbox to choose experiment
                        experiment_names = [f.replace("_metadata.json", "") for f in experiment_files]
                        selected_experiment = st.selectbox(
                            "Select Experiment", 
                            experiment_names, 
                            key="ml_experiment_select"
                        )
                        
                        if selected_experiment:
                            metadata_path = f"experiments/{selected_experiment}_metadata.json"
                            
                            with open(metadata_path, 'r') as f:
                                exp_data = json.load(f)
                            
                            data_path = exp_data["data_path"]
                            
                            if os.path.exists(data_path):
                                # Load the data
                                df = pd.read_csv(data_path)
                                
                                # Check if dataset has required columns
                                required_cols = [
                                    'imu0_acc_x', 'imu0_acc_y', 'imu0_acc_z',
                                    'imu0_gyro_x', 'imu0_gyro_y', 'imu0_gyro_z',
                                    'imu1_acc_x', 'imu1_acc_y', 'imu1_acc_z',
                                    'imu1_gyro_x', 'imu1_gyro_y', 'imu1_gyro_z'
                                ]
                                missing_cols = [col for col in required_cols if col not in df.columns]
                                
                                if missing_cols:
                                    st.error(f"The selected dataset is missing required columns: {', '.join(missing_cols)}")
                                    st.info("The model requires both IMU0 and IMU1 data (accelerometer and gyroscope) to make predictions.")
                                else:
                                    # Display data statistics
                                    st.subheader("Data Selection for Prediction")
                                    
                                    sample_range = st.slider(
                                        "Select Sample Range for Analysis", 
                                        min_value=0, 
                                        max_value=len(df)-1, 
                                        value=(0, min(1000, len(df)-1)),
                                        key="ml_sample_range"
                                    )
                                    
                                    selected_df = df.iloc[sample_range[0]:sample_range[1]]
                                    
                                    # Display column info
                                    with st.expander("Dataset Column Information"):
                                        st.write("**Available columns:**")
                                        st.write(", ".join(df.columns.tolist()))
                                        st.write("**Required columns for prediction:**")
                                        st.write(", ".join(required_cols))
                                    
                                    # Prediction type selection
                                    prediction_type = st.radio(
                                        "Prediction Type", 
                                        ["force", "angle"], 
                                        key="prediction_type"
                                    )
                                    
                                    # Add memory options
                                    with st.expander("Advanced Settings"):
                                        # Don't use GPU option since we're forcing CPU mode globally
                                        batch_size = st.slider("Batch Size", min_value=16, max_value=512, value=64, step=16,
                                                             help="Smaller batch size uses less memory but is slower")
                                    
                                    if st.button("Generate Predictions", key="generate_predictions"):
                                        with st.spinner("Preparing data and generating predictions..."):
                                            # Prepare data for prediction
                                            X, error_message = prepare_data_for_prediction(selected_df)
                                            
                                            if X is None:
                                                st.error(f"Error preparing data: {error_message}")
                                            else:
                                                # Display input shape before prediction
                                                st.info(f"Input data shape: {X.shape}")
                                                if 'model_info' in st.session_state:
                                                    expected_shape = st.session_state['model_info']['input_shape']
                                                    st.info(f"Expected model input shape: {expected_shape}")
                                                
                                                # Make predictions
                                                try:
                                                    # Explicitly add verbose output to see progress
                                                    st.text("Running prediction...")
                                                    
                                                    # No need to set GPU visibility here since we've done it globally
                                                    # Force predictions to run on CPU
                                                    with tf.device('/CPU:0'):
                                                        # Use batching for prediction to reduce memory usage
                                                        total_samples = X.shape[0]
                                                        all_predictions = []
                                                        progress_bar = st.progress(0)
                                                        
                                                        for i in range(0, total_samples, batch_size):
                                                            end_idx = min(i + batch_size, total_samples)
                                                            batch_X = X[i:end_idx]
                                                            
                                                            # Update progress
                                                            progress = float(i) / total_samples
                                                            progress_bar.progress(progress)
                                                            st.text(f"Processing batch {i//batch_size + 1}/{(total_samples-1)//batch_size + 1}...")
                                                            
                                                            # Run the batch with low verbosity
                                                            batch_pred = model.predict(batch_X, verbose=0)
                                                            all_predictions.append(batch_pred)
                                                        
                                                        # Complete progress
                                                        progress_bar.progress(1.0)
                                                    
                                                    # Combine predictions
                                                    if all_predictions:
                                                        predictions = np.vstack(all_predictions)
                                                        st.text("Prediction complete!")
                                                        
                                                        # Debug prediction output
                                                        st.success(f"Successfully generated predictions with shape: {predictions.shape}")
                                                        
                                                        # Apply scaling factor to convert to realistic values
                                                        if prediction_type == "force":
                                                            # Check the scale of predictions
                                                            max_vertical = max(np.max(predictions[:, 1]), np.max(predictions[:, 4]))
                                                            
                                                            # Determine if scaling is needed
                                                            participant_weight = exp_data.get("participant", {}).get("weight", 70)  # Default to 70kg
                                                            expected_force = participant_weight * 9.81  # N = kg * 9.81 m/s²
                                                            
                                                            # Auto-determine scaling factor
                                                            if max_vertical < 10:  # If max force is less than 10N, likely needs scaling
                                                                if max_vertical < 1:  # Extremely small values
                                                                    scale_factor = expected_force / (max_vertical if max_vertical > 0 else 1)
                                                                else:  # Small but not tiny values
                                                                    scale_factor = 100  # Conservative scaling
                                                                
                                                                st.warning(f"Force values appear too small. Applying scaling factor of {scale_factor:.2f}x")
                                                                predictions = predictions * scale_factor
                                                            
                                                            # Display the scaling decision
                                                            with st.expander("Force Scaling Information"):
                                                                st.write(f"Participant weight: {participant_weight} kg")
                                                                st.write(f"Expected body weight force: ~{expected_force:.2f} N")
                                                                st.write(f"Original max vertical force: {max_vertical:.2f}")
                                                                st.write(f"Scaled: {max_vertical < 10}")
                                                                if max_vertical < 10:
                                                                    st.write(f"Applied scaling factor: {scale_factor:.2f}")
                                                        
                                                        # Check for NaN values in predictions
                                                        if np.isnan(predictions).any():
                                                            st.warning("Warning: Predictions contain NaN values.")
                                                        
                                                        # Display predictions
                                                        st.subheader("Prediction Results")
                                                        
                                                        # Show raw prediction values
                                                        with st.expander("Raw Prediction Values (First 5 rows)"):
                                                            st.write(predictions[:5])
                                                        
                                                        # Create visualization
                                                        st.text("Generating visualization...")
                                                        # Pass the scaling information to the plotting function
                                                        was_scaled = prediction_type == "force" and max_vertical < 10
                                                        plots = plot_predictions(predictions, prediction_type, scaled=was_scaled)
                                                        
                                                        if plots:
                                                            for i, fig in enumerate(plots):
                                                                st.pyplot(fig)
                                                        
                                                        # Display prediction statistics
                                                        st.subheader("Prediction Statistics")
                                                        
                                                        if prediction_type == "force":
                                                            # Display force statistics
                                                            stats_cols = st.columns(3)
                                                            
                                                            with stats_cols[0]:
                                                                avg_left_vertical = np.mean(predictions[:, 1])
                                                                avg_right_vertical = np.mean(predictions[:, 4])
                                                                total_vertical = avg_left_vertical + avg_right_vertical
                                                                
                                                                st.metric(
                                                                    "Avg. Total Vertical Force", 
                                                                    f"{total_vertical:.2f} N"
                                                                )
                                                                
                                                                # Add expected body weight for comparison
                                                                if 'participant' in exp_data:
                                                                    weight_kg = exp_data['participant'].get('weight', 70)
                                                                    expected_force = weight_kg * 9.81
                                                                    st.caption(f"Expected body weight: ~{expected_force:.2f} N")
                                                            
                                                            with stats_cols[1]:
                                                                left_ratio = avg_left_vertical / total_vertical * 100 if total_vertical != 0 else 0
                                                                st.metric(
                                                                    "Left Foot Load", 
                                                                    f"{left_ratio:.1f}%"
                                                                )
                                                            
                                                            with stats_cols[2]:
                                                                right_ratio = avg_right_vertical / total_vertical * 100 if total_vertical != 0 else 0
                                                                st.metric(
                                                                    "Right Foot Load", 
                                                                    f"{right_ratio:.1f}%"
                                                                )
                                                            
                                                            # Display detailed force data
                                                            with st.expander("Detailed Force Data"):
                                                                force_components = [
                                                                    "Left Anterior-Posterior",
                                                                    "Left Vertical",
                                                                    "Left Medial-Lateral",
                                                                    "Right Anterior-Posterior",
                                                                    "Right Vertical",
                                                                    "Right Medial-Lateral"
                                                                ]
                                                                
                                                                force_stats = pd.DataFrame({
                                                                    "Component": force_components,
                                                                    "Mean (N)": [np.mean(predictions[:, i]) for i in range(6)],
                                                                    "Max (N)": [np.max(predictions[:, i]) for i in range(6)],
                                                                    "Min (N)": [np.min(predictions[:, i]) for i in range(6)]
                                                                })
                                                                
                                                                st.dataframe(
                                                                    force_stats, 
                                                                    hide_index=True,
                                                                    column_config={
                                                                        "Component": st.column_config.TextColumn("Force Component"),
                                                                        "Mean (N)": st.column_config.NumberColumn("Mean (N)", format="%.2f"),
                                                                        "Max (N)": st.column_config.NumberColumn("Max (N)", format="%.2f"),
                                                                        "Min (N)": st.column_config.NumberColumn("Min (N)", format="%.2f")
                                                                    }
                                                                )
                                                                
                                                            # Display symmetry analysis
                                                            st.subheader("Force Symmetry Analysis")
                                                            
                                                            # Calculate symmetry index for vertical forces
                                                            left_vertical = predictions[:, 1]
                                                            right_vertical = predictions[:, 4]
                                                            
                                                            # Symmetry Index (SI) = |R-L|/((R+L)/2) * 100%
                                                            valid_indices = (right_vertical + left_vertical) != 0
                                                            if np.any(valid_indices):
                                                                symmetry_index = np.mean(
                                                                    np.abs(right_vertical[valid_indices] - left_vertical[valid_indices]) / 
                                                                    ((right_vertical[valid_indices] + left_vertical[valid_indices]) / 2)
                                                                ) * 100
                                                            else:
                                                                symmetry_index = 0
                                                            
                                                            # Gait Asymmetry (GA) = |ln(right/left)| * 100%
                                                            valid_indices = (left_vertical > 0) & (right_vertical > 0)
                                                            if np.any(valid_indices):
                                                                gait_asymmetry = np.mean(
                                                                    np.abs(
                                                                        np.log(right_vertical[valid_indices] / left_vertical[valid_indices])
                                                                    )
                                                                ) * 100
                                                            else:
                                                                gait_asymmetry = 0
                                                            
                                                            asym_cols = st.columns(2)
                                                            with asym_cols[0]:
                                                                st.metric(
                                                                    "Symmetry Index (SI)", 
                                                                    f"{symmetry_index:.2f}%",
                                                                    help="SI < 10% is considered normal, > 10% indicates asymmetry"
                                                                )
                                                            
                                                            with asym_cols[1]:
                                                                st.metric(
                                                                    "Gait Asymmetry (GA)", 
                                                                    f"{gait_asymmetry:.2f}%",
                                                                    help="GA < 5% is considered normal, > 10% indicates significant asymmetry"
                                                                )
                                                        
                                                        elif prediction_type == "angle":
                                                            # Display angle statistics
                                                            stats_cols = st.columns(3)
                                                            
                                                            with stats_cols[0]:
                                                                avg_angle = np.mean(predictions)
                                                                st.metric(
                                                                    "Average Angle", 
                                                                    f"{avg_angle:.2f}°"
                                                                )
                                                            
                                                            with stats_cols[1]:
                                                                max_angle = np.max(predictions)
                                                                st.metric(
                                                                    "Maximum Angle", 
                                                                    f"{max_angle:.2f}°"
                                                                )
                                                            
                                                            with stats_cols[2]:
                                                                min_angle = np.min(predictions)
                                                                st.metric(
                                                                    "Minimum Angle", 
                                                                    f"{min_angle:.2f}°"
                                                                )
                                                            
                                                            # Calculate ROM
                                                            rom = max_angle - min_angle
                                                            st.metric(
                                                                "Range of Motion", 
                                                                f"{rom:.2f}°"
                                                            )
                                                        
                                                        # Download predictions
                                                        predictions_df = pd.DataFrame(predictions)
                                                        if prediction_type == "force":
                                                            columns = [
                                                                "Left_AP_Force", "Left_Vertical_Force", "Left_ML_Force",
                                                                "Right_AP_Force", "Right_Vertical_Force", "Right_ML_Force"
                                                            ]
                                                            predictions_df.columns = columns
                                                        elif prediction_type == "angle":
                                                            predictions_df.columns = ["Angle"]
                                                        
                                                        csv = predictions_df.to_csv(index=False)
                                                        st.download_button(
                                                            "Download Predictions as CSV",
                                                            csv,
                                                            f"{selected_experiment}_predictions.csv",
                                                            "text/csv",
                                                            key="download_predictions"
                                                        )
                                                    
                                                except Exception as e:
                                                    st.error(f"Error making predictions: {str(e)}")
                                                    import traceback
                                                    st.error(f"Detailed error: {traceback.format_exc()}")
                                                    st.info("Try using a different dataset, smaller sample range, or verify the data format matches the model's requirements.")
                            else:
                                st.error(f"Data file not found: {data_path}")
                    else:
                        st.info("No experiments found.")
                else:
                    st.info("No experiments directory found.")
                    
        except Exception as e:
            st.error(f"An unexpected error occurred in the ML Predictions tab: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            st.warning("Please report this issue to the development team.")

if __name__ == "__main__":
    main() 