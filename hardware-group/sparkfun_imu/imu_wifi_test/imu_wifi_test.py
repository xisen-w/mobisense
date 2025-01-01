from flask import Flask, request, jsonify
import csv
import datetime

app = Flask(__name__)

# File to store data
data_file = "arduino_data.csv"

# Initialize CSV file with headers if it doesn't exist
def init_csv():
    try:
        with open(data_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Check if the file is empty (no headers yet)
            if file.tell() == 0:
                writer.writerow([
                    "acceleration_x", "acceleration_y", "acceleration_z",
                    "gyro_x", "gyro_y", "gyro_z", "timestamp"
                ])  # Add headers
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
            with open(data_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    data.get('acceleration_x', None),
                    data.get('acceleration_y', None),
                    data.get('acceleration_z', None),
                    data.get('gyro_x', None),
                    data.get('gyro_y', None),
                    data.get('gyro_z', None),
                    timestamp
                ])

            print(f"Received data: {data}")
            return jsonify({"status": "success"}), 200
        else:
            return jsonify({"status": "error", "message": "No data received"}), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    init_csv()  # Initialize the CSV file with headers when the server starts
    app.run(host='0.0.0.0', port=8000)
