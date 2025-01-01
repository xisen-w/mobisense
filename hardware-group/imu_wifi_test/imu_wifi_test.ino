#include "ICM_20948.h" // Click here to get the library: http://librarymanager/All#SparkFun_ICM_20948_IMU
#include <WiFiS3.h>    // For Wi-Fi connectivity
#include <ArduinoHttpClient.h> // For HTTP requests

// Wi-Fi credentials and server details
const char* ssid = "FASTWEB-EE2AD9";       // Replace with your Wi-Fi network name
const char* password = "1K12WH46G7";       // Replace with your Wi-Fi password
const char* server = "192.168.1.174";      // Replace with your server's IP address
int port = 8000;                           // Replace with your server's port
const char* path = "/api/data";            // Replace with your API endpoint

WiFiClient wifi;                // Wi-Fi client
HttpClient client(wifi, server, port); // HTTP client

#define SERIAL_PORT Serial
#define WIRE_PORT Wire1  // I2C communication
#define AD0_VAL 1        // Set I2C address based on the ADR jumper

ICM_20948_I2C myICM; // Use I2C for communication with the sensor

void setup()
{
  SERIAL_PORT.begin(115200);
  while (!SERIAL_PORT)
  {
  };

  WIRE_PORT.begin();
  WIRE_PORT.setClock(400000); // Set I2C clock speed

  // Initialize the IMU sensor
  bool initialized = false;
  while (!initialized)
  {
    myICM.begin(WIRE_PORT, AD0_VAL); // Initialize I2C communication

    SERIAL_PORT.print(F("Initialization of the sensor returned: "));
    SERIAL_PORT.println(myICM.statusString());
    if (myICM.status != ICM_20948_Stat_Ok)
    {
      SERIAL_PORT.println("Trying again...");
      delay(500);
    }
    else
    {
      initialized = true;
    }
  }

  // Connect to Wi-Fi
  SERIAL_PORT.println("Connecting to Wi-Fi...");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED)
  {
    delay(1000);
    SERIAL_PORT.println("Connecting...");
  }
  SERIAL_PORT.println("Connected to Wi-Fi");
}

void loop()
{
  if (myICM.dataReady())
  {
    myICM.getAGMT(); // Update sensor data

    // Prepare JSON payload
    String payload = "{";
    payload += "\"acceleration_x\":" + String(myICM.accX(), 2) + ",";
    payload += "\"acceleration_y\":" + String(myICM.accY(), 2) + ",";
    payload += "\"acceleration_z\":" + String(myICM.accZ(), 2) + ",";
    payload += "\"gyro_x\":" + String(myICM.gyrX(), 2) + ",";
    payload += "\"gyro_y\":" + String(myICM.gyrY(), 2) + ",";
    payload += "\"gyro_z\":" + String(myICM.gyrZ(), 2);
    payload += "}";

    // Send HTTP POST request
    client.beginRequest();
    client.post(path);
    client.sendHeader("Content-Type", "application/json");
    client.sendHeader("Content-Length", payload.length());
    client.beginBody();
    client.print(payload);
    client.endRequest();

    // Check server response
    int statusCode = client.responseStatusCode();
    String response = client.responseBody();

    SERIAL_PORT.print("Status Code: ");
    SERIAL_PORT.println(statusCode);
    SERIAL_PORT.print("Response: ");
    SERIAL_PORT.println(response);

    delay(1000); // Send data every second
  }
  else
  {
    SERIAL_PORT.println("Waiting for data");
    delay(500);
  }
}

// Below here are helper functions to print the data nicely!

void printPaddedInt16b(int16_t val)
{
  if (val > 0)
  {
    SERIAL_PORT.print(" ");
    if (val < 10000)
    {
      SERIAL_PORT.print("0");
    }
    if (val < 1000)
    {
      SERIAL_PORT.print("0");
    }
    if (val < 100)
    {
      SERIAL_PORT.print("0");
    }
    if (val < 10)
    {
      SERIAL_PORT.print("0");
    }
  }
  else
  {
    SERIAL_PORT.print("-");
    if (abs(val) < 10000)
    {
      SERIAL_PORT.print("0");
    }
    if (abs(val) < 1000)
    {
      SERIAL_PORT.print("0");
    }
    if (abs(val) < 100)
    {
      SERIAL_PORT.print("0");
    }
    if (abs(val) < 10)
    {
      SERIAL_PORT.print("0");
    }
  }
  SERIAL_PORT.print(abs(val));
}

void printFormattedFloat(float val, uint8_t leading, uint8_t decimals)
{
  float aval = abs(val);
  if (val < 0)
  {
    SERIAL_PORT.print("-");
  }
  else
  {
    SERIAL_PORT.print(" ");
  }
  for (uint8_t indi = 0; indi < leading; indi++)
  {
    uint32_t tenpow = 0;
    if (indi < (leading - 1))
    {
      tenpow = 1;
    }
    for (uint8_t c = 0; c < (leading - 1 - indi); c++)
    {
      tenpow *= 10;
    }
    if (aval < tenpow)
    {
      SERIAL_PORT.print("0");
    }
    else
    {
      break;
    }
  }
  if (val < 0)
  {
    SERIAL_PORT.print(-val, decimals);
  }
  else
  {
    SERIAL_PORT.print(val, decimals);
  }
}