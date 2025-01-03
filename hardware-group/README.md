# Hardware Team
## Setup
Our tentative setup:
- Board: Arduino UNO R4 WiFi
- EMG sensor: Grove EMG detector v1.1
- IMU sensor: SparkFun 9-DoF ICM-20948 (with breakout board)
- I2C Multiplexor: SparkFun Qwiic MUX v1.1

## EMG
You don't need any libraries to run the test script.

## IMU
To run the IMU test scripts via Arduino IDE, you'll need the following libraries enabled:
- SparkFun 9DoF IMU Breakout - ICM 20948 `v1.3.0`
- SparkFun I2C Mux `v1.0.3`