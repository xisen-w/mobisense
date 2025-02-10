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
UDP_IP = "0.0.0.0"         # Listen on all network interfaces
UDP_PORT = 8000
BUFFER_SIZE = 4096          # Maximum size of the payload

# metadata
NUMBER_OF_SENSORS = 2

# File to store data
data_folder = os.path.dirname(__file__) + "/data"
os.makedirs(data_folder, exist_ok=True)
file_path = data_folder + datetime.datetime.now().strftime("/%Y-%m-%d_%H-%M-%S.csv")


# Initialize CSV file with headers if it doesn't exist
def init_csv():
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
def save_data_to_csv(imu_data):
    """Save data to CSV file

    Args:
        imu_data: JSON header data from the Arduino
    """
    try:
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            timestamp = datetime.datetime.now().isoformat()
            rows = []
            for i in range(NUMBER_OF_SENSORS):
                imu = imu_data[i]
                rows += [
                    imu.get('acceleration').get('x', None),
                    imu.get('acceleration').get('y', None),
                    imu.get('acceleration').get('z', None),
                    imu.get('gyro').get('x', None),
                    imu.get('gyro').get('y', None),
                    imu.get('gyro').get('z', None),
                    imu.get('orientation').get('roll', None),
                    imu.get('orientation').get('pitch', None),
                    imu.get('orientation').get('yaw', None),
                    timestamp
                ]
            writer.writerow(rows)
    except Exception as e:
        print(f"Error saving data to CSV file: {e}")


# udp server
def udp_server():
    "Receives and processes data sent by the Arduino over UDP"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"UDP server listening on {UDP_IP}:{UDP_PORT}")

    while True:
        try:
            # Receive raw binary data from the Arduino and parse json data
            data, addr = sock.recvfrom(BUFFER_SIZE)
            decoded_imu_data = json.loads(data.decode("utf-8"))
            print(f"Received data from {addr}: {decoded_imu_data}")

            # Save data to CSV file
            if "imu_data" in decoded_imu_data:
                save_data_to_csv(decoded_imu_data["imu_data"])
        except Exception as e:
            print(f"Error receiving data: {e}")


if __name__ == '__main__':
    init_csv()
    udp_server()
