import socket
import json
import csv
import datetime
import os


# Get the local IP address of the active network interface
def get_local_ip():
    """Automatically find the local IP address of the active network interface."""
    try:
        # Create a dummy socket connection to an external address (Google DNS)
        temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        temp_sock.connect(("8.8.8.8", 80))  # Doesn't actually send data
        local_ip = temp_sock.getsockname()[0]  # Get local IP used for the connection
        temp_sock.close()
        return local_ip
    except Exception as e:
        print(f"Error getting local IP: {e}")
        return "127.0.0.1"  # Fallback to localhost if detection fails


# UDP server configuration
UDP_IP = get_local_ip()     # Listen on local machine's IP address
UDP_PORT = 8000
BUFFER_SIZE = 256           # Maximum size of the payload

# imu metadata
NUMBER_OF_SENSORS = 2
EXPECTED_VALUES = 9         # Each IMU has 9 values

# File to store data
data_folder = os.path.dirname(__file__) + "/data"
os.makedirs(data_folder, exist_ok=True)


# Initialize CSV file with headers if it doesn't exist
def init_csv(file_path):
    try:
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Check if the file is empty (no headers yet)
            if file.tell() == 0:
                headers = []
                for i in range(NUMBER_OF_SENSORS):
                    imu = f"imu{i}"
                    headers += [
                        f"{imu}_acc_x", f"{imu}_acc_y", f"{imu}_acc_z",
                        f"{imu}_gyro_x", f"{imu}_gyro_y", f"{imu}_gyro_z",
                        f"{imu}_roll", f"{imu}_pitch", f"{imu}_yaw",
                        f"{imu}_timestamp"
                    ]
                writer.writerow(headers)  # Add headers
    except Exception as e:
        print(f"Error initializing CSV file: {e}")


# Save data to csv file
def save_data_to_csv(imu_data, file_path):
    """Save data to CSV file

    Args:
        imu_data: pased csv data from the Arduino
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                timestamp = datetime.datetime.now().isoformat()
                for i in range(NUMBER_OF_SENSORS):
                    imu_data.insert(10 * i + EXPECTED_VALUES, timestamp)
                writer.writerow(imu_data)
        except Exception as e:
            print(f"Error saving data to CSV file: {e}")


# udp server
def udp_server():
    "Receives and processes data sent by the Arduino over UDP"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"UDP server listening on {UDP_IP}:{UDP_PORT}")

    file_path = None

    while True:
        try:
            # Receive raw binary data from the Arduino and parse csv data
            data, addr = sock.recvfrom(BUFFER_SIZE)
            data = data.decode("utf-8").strip()
            print(f"Received data from {addr}: {data}")
            imu_data = data.split(",")

            # Check for start event
            if imu_data == ["start"]:
                file_path = data_folder + datetime.datetime.now().strftime("/%Y-%m-%d_%H-%M-%S.csv")
                init_csv(file_path)
                continue
            
            if file_path is not None:
                if imu_data == ["starting..."]:
                    continue

                expected_values = NUMBER_OF_SENSORS * EXPECTED_VALUES
                if len(imu_data) != expected_values:
                    print(f"Warning: Expected {expected_values} values, got {len(imu_data)}")
                    continue  # Skip bad packets

                parsed_data = [float(v) for v in imu_data]
                save_data_to_csv(parsed_data, file_path)

        except Exception as e:
            print(f"Error processing UDP data: {e}")


if __name__ == '__main__':
    udp_server()
