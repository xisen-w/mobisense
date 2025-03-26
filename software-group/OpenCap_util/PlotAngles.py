import pandas as pd
import matplotlib.pyplot as plt
import os

# Directory containing the .mot files (adjust if needed)
kinematics_folder = 'OpenCapData_015964a1-b2ef-4c73-9ff7-d0b6be3a836f/OpenSimData/Kinematics'

# List of .mot files
mot_files = [
    'limping1.mot',
    'limping2.mot',
    'limping3.mot',
    'walk1.mot',
    'walk2.mot',
    'walk3.mot'
]

# Function to read and process .mot file
def read_mot_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the index of the line where data starts
    data_start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith('time') or line.strip().startswith('time\t'):
            data_start_idx = idx
            break

    if data_start_idx is None:
        raise ValueError(f"Could not find data header in {file_path}")

    # Extract column names and data
    column_names = lines[data_start_idx].strip().split()
    data = [list(map(float, line.strip().split())) for line in lines[data_start_idx+1:] if line.strip()]

    df = pd.DataFrame(data, columns=column_names)
    return df

# Plotting function
def plot_ankle_angles(df, filename, output_folder='.'):
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['ankle_angle_r'], label='Ankle Angle Right')
    plt.plot(df['time'], df['ankle_angle_l'], label='Ankle Angle Left')
    plt.xlabel('Time (s)')
    plt.ylabel('Ankle Angle (degrees)')
    plt.title(f'Ankle Angles Over Time: {filename}')
    plt.legend()
    plt.grid(True)

    # Save the plot as a PNG
    output_path = os.path.join(output_folder, f'{filename}_ankle_angles.png')
    plt.savefig(output_path)
    plt.close()
    print(f'Saved plot to {output_path}')

# Iterate over the files and generate plots
for mot_file in mot_files:
    file_path = os.path.join(kinematics_folder, mot_file)
    try:
        df = read_mot_file(file_path)
        plot_ankle_angles(df, mot_file.split('.')[0])
    except Exception as e:
        print(f"Error processing {mot_file}: {e}")
