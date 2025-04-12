import streamlit as st
import pandas as pd
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from experiments import Participant, IMU_Experiment_Setup

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
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Time array
    time_values = pd.to_datetime(df['imu0_timestamp'].iloc[start_idx:end_idx])
    time_seconds = [(t - time_values.iloc[0]).total_seconds() for t in time_values]
    
    # Plot angle data
    ax.plot(time_seconds, df['dorsiflexion_angle'].iloc[start_idx:end_idx], label='Dorsiflexion Angle')
    ax.set_title('Dorsiflexion Angle')
    ax.set_ylabel('Angle (degrees)')
    ax.set_xlabel('Time (s)')
    ax.grid(True)
    
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
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Time array
    time_col = f'imu{imu_num}_timestamp'
    time_values = pd.to_datetime(df[time_col].iloc[start_idx:end_idx])
    time_seconds = [(t - time_values.iloc[0]).total_seconds() for t in time_values]
    
    # Plot orientation data
    ax.plot(time_seconds, df[f'imu{imu_num}_roll'].iloc[start_idx:end_idx], label='Roll')
    ax.plot(time_seconds, df[f'imu{imu_num}_pitch'].iloc[start_idx:end_idx], label='Pitch')
    ax.plot(time_seconds, df[f'imu{imu_num}_yaw'].iloc[start_idx:end_idx], label='Yaw')
    
    ax.set_title(f'IMU {imu_num} Orientation')
    ax.set_ylabel('Angle (degrees)')
    ax.set_xlabel('Time (s)')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    return fig

def main():
    st.title("MobiSense Experiment Setup")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Data Upload & Experiment Setup", "View Experiments", "Data Visualization"])
    
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
                        
                        imu_selection = st.radio("Select IMU", [0, 1])
                        
                        sample_range = st.slider(
                            "Select Sample Range", 
                            min_value=0, 
                            max_value=len(df)-1, 
                            value=(0, min(500, len(df)-1))
                        )
                        
                        # Plot IMU data
                        st.subheader(f"IMU {imu_selection} Data")
                        imu_fig = plot_imu_data(df, imu_num=imu_selection, start_idx=sample_range[0], end_idx=sample_range[1])
                        st.pyplot(imu_fig)
                        
                        # Plot orientation data (roll, pitch, yaw)
                        orientation_columns = [f'imu{imu_selection}_roll', f'imu{imu_selection}_pitch', f'imu{imu_selection}_yaw']
                        if all(col in df.columns for col in orientation_columns):
                            st.subheader(f"IMU {imu_selection} Orientation (Roll, Pitch, Yaw)")
                            orientation_fig = plot_orientation_data(df, imu_num=imu_selection, start_idx=sample_range[0], end_idx=sample_range[1])
                            st.pyplot(orientation_fig)
                        else:
                            st.info(f"Orientation data (roll, pitch, yaw) not available for IMU {imu_selection}")
                        
                        # Plot angle data if available
                        if has_dorsiflexion_angle:
                            st.subheader("Dorsiflexion Angle Data")
                            angle_fig = plot_angle_data(df, start_idx=sample_range[0], end_idx=sample_range[1])
                            st.pyplot(angle_fig)
                        else:
                            st.warning("No dorsiflexion angle data available in this dataset.")
                    else:
                        st.error(f"Data file not found: {data_path}")
            else:
                st.info("No experiments found.")
        else:
            st.info("No experiments directory found.")

if __name__ == "__main__":
    main() 