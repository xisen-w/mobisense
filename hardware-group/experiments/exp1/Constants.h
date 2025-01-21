#ifndef CONSTANTS_H
#define CONSTANTS_H

// Wi-Fi settings
const char* SSID = "SHIL-WIFI";
const char* PASSWORD = "bhqcclcvjcvk";

// Server settings
const char* SERVER = "10.130.149.170";  // Boting's Laptop
const int PORT = 8000;
const char* API_PATH = "/api/data";

// Arduino Board settings
#define SERIAL_PORT Serial
#define WIRE_PORT Wire1         // Qwiic I2C communication

// IMU settings
#define AD0_VAL 1               // The value of the last bit of the IMU I2C address. Defaults to 1.
#define NUMBER_OF_SENSORS 2     // Number of IMUs connected to the Qwiic Mux
#define I2C_CLOCK_SPEED 400000  // I2C clock speed in Hz
#define MUX_ADDR 0x70           // Default I2C address of the Qwiic Mux
#define SAMPLE_RATE 1         // Sample rate in Hz

#endif