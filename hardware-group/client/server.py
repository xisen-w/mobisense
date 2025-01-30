from flask import Flask, request, jsonify
import csv
import datetime
import os

app = Flask(__name__)

# metadata
NUMBER_OF_SENSORS = 2

# File to store data
data_folder = os.getcwd() + "/data"
os.makedirs(data_folder, exist_ok=True)
file_path = data_folder + datetime.datetime.now().strftime("/%Y-%m-%d_%H-%M-%S.csv")


# Initialize CSV file with headers if it doesn't exist
def init_csv():
    try:
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Check if the file is empty (no headers yet)
            if file.tell() == 0:
                rows = []
                for i in range(NUMBER_OF_SENSORS):
                    imu = "imu" + str(i)
                    rows.append(imu + "_acc_x")
                    rows.append(imu + "_acc_y")
                    rows.append(imu + "_acc_z")
                    rows.append(imu + "_gyro_x")
                    rows.append(imu + "_gyro_y")
                    rows.append(imu + "_gyro_z")
                    rows.append(imu + "_roll")
                    rows.append(imu + "_pitch")
                    rows.append(imu + "_yaw")
                    rows.append(imu + "_timestamp")
                writer.writerow(rows)  # Add headers

    except Exception as e:
        print(f"Error initializing CSV file: {e}")


@app.route('/api/data', methods=['POST'])
def receive_data():
    try:
        # Get JSON payload from Arduino
        data = request.json
        if data:
            # Add a timestamp to the received data
            timestamp = datetime.datetime.now().isoformat()

            # Append data to the CSV file
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                imu_data = data.get('imu_data', None)
                rows = []
                for i in range(NUMBER_OF_SENSORS):
                    imu = imu_data[i]
                    rows.append(imu.get('acceleration').get('x', None))
                    rows.append(imu.get('acceleration').get('y', None))
                    rows.append(imu.get('acceleration').get('z', None))
                    rows.append(imu.get('gyro').get('x', None))
                    rows.append(imu.get('gyro').get('y', None))
                    rows.append(imu.get('gyro').get('z', None))
                    rows.append(imu.get('roll'))
                    rows.append(imu.get('pitch'))
                    rows.append(imu.get('yaw'))
                    rows.append(timestamp)
                writer.writerow(rows)

            print(f"Received data: {imu_data}")
            return jsonify({"status": "success"}), 200
        else:
            return jsonify({"status": "error", "message": "No data received"}), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    init_csv()  # Initialize the CSV file with headers when the server starts
    app.run(host='0.0.0.0', port=8000)
