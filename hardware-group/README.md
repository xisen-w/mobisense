# Hardware Team
## Setup
Our tentative setup:
- Board: Arduino UNO R4 WiFi
- EMG sensor: Grove EMG detector v1.1
- IMU sensor: SparkFun 9-DoF ICM-20948 (with breakout board)
- I2C Multiplexor: SparkFun Qwiic MUX v1.1

## Configuration
### EMG
You don't need any libraries to run the test script.

### IMU
To run the IMU test scripts via Arduino IDE, you'll need the following libraries enabled:
- `SparkFun 9DoF IMU Breakout - ICM 20948`, `v1.3.0`
- `SparkFun I2C Mux`, `v1.0.3`
- `ArduinoHttpClient`, `v0.6.1`

## Experiment
Once all test scripts are successfully run, you can proceed to run the actual client.

1. When using the Qwiic MUX, make sure you start with port 0 and follow the ascending numerical order (1, 2, ...).

2. Flashing `client.ino` to set up the client on Arduino.
3. `cd <path to repo>/mobisense/hardware-group/client/`
4. `python server.py`

## Trouble Shooting
If no data is received on the server,
- try restarting Arduino by pressing the button
- try terminating and rerunning `server.py`
- check your wifi connection by running `sparkfun_imu/imu_wifi_test` with your Arduino plugged in your laptop

## Inverse Kinematics Script  
This script calculates joint angles in the sagittal plane using IMU sensor data. It helps analyze lower limb movement.  

Note Digital Motion Processor (DMP) fusion algorithm requires ~45 seconds to stabilize. Wait 45 seconds before recording data.

Assumptions  
- IMU reference axes must be aligned:
  - X-axis points away from the treadmill.  
  - Z-axis points towards the ceiling.  
- IMU placement on the limb: 
  - X-axis extends away from the treadmill.  
  - Z-axis points away from the body.  
