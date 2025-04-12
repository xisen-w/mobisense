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

def main():
    st.title("MobiSense Experiment Setup")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Data Upload & Experiment Setup", "View Experiments", "Data Visualization", "Gait Analytics"])
    
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

if __name__ == "__main__":
    main() 