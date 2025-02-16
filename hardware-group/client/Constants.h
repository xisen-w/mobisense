#ifndef CONSTANTS_H
#define CONSTANTS_H

// Wi-Fi settings
// const char* SSID = "bt_iPhone";
// const char* PASSWORD = "hahahaha";
// const char* SSID = "iPhone di francesco (2)";
// const char* PASSWORD = "swag1234";
// const char* SSID = "SHIL-WIFI";
// const char* PASSWORD = "bhqcclcvjcvk";
const char* SSID = "Hertford";
const char* PASSWORD = "HaIhRLwreEnp";

// Server settings
const char* SERVER = "10.133.176.13";  // Boting's Laptop
const int PORT = 8000;
const char* API_PATH = "/api/data";

// Arduino UDP settings
#define LOCAL_UDP_PORT 8001     // Local port for sending UDP packet

// Arduino Board settings
#define SERIAL_PORT Serial
#define WIRE_PORT Wire1         // Qwiic I2C communication

// IMU settings
#define AD0_VAL 1               // The value of the last bit of the IMU I2C address. Defaults to 1.
#define NUMBER_OF_SENSORS 2     // Number of IMUs connected to the Qwiic Mux
#define I2C_CLOCK_SPEED 400000  // I2C clock speed in Hz
#define MUX_ADDR 0x70           // Default I2C address of the Qwiic Mux
#define SAMPLE_RATE 100         // Sample rate in Hz

// LED Matrix settings
const uint32_t startFrame[] =
{
    0xFFFFFFFF,
    0xFFFFFFFF,
    0xFFFFFFFF
};
const uint32_t dangerFrame[] =
{
	0x400a015,
	0x1502082,
	0x484047fc
};

#endif