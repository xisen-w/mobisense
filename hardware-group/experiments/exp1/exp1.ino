#include <Wire.h>

#include <WiFiS3.h>                             // For Wi-Fi connectivity
#include <ArduinoHttpClient.h>                  // For HTTP requests

#include <ICM_20948.h>                          // For IMUs
#include <SparkFun_I2C_Mux_Arduino_Library.h>   // For MUX

#include "Constants.h"  // Pre-defined constants
#include "Helpers.h"    // Helper functions to print results nicely

WiFiClient WIFI;                        // Wi-Fi client
HttpClient client(WIFI, SERVER, PORT);  // HTTP client

QWIICMUX myMux;         // Create instance of the Qwiic Mux class
ICM_20948_I2C **myICM;  // Create pointer to a set of pointers to the sensor class

void setup()
{
  SERIAL_PORT.begin(115200);
  while (!SERIAL_PORT)
  {
    // hang
  };


  /* ######################### Initialising MUX ######################### */
  SERIAL_PORT.println("Initialising IMUs via Qwiic I2C MUX...");

  // Open Wire1() for Qwiix I2C
  WIRE_PORT.begin();
  WIRE_PORT.setClock(I2C_CLOCK_SPEED); // Set I2C clock speed

  // Check MUX connection
  if (myMux.begin(MUX_ADDR, WIRE_PORT) == false)
  {
    SERIAL_PORT.println("Mux not detected. Reset Uno Board to re-detect. Freezing...");
    while(1);
  }
  SERIAL_PORT.println("Mux detected. Initialising IMUs...");

  // Check Port
  byte currentPortNumber = myMux.getPort();
  Serial.print("CurrentPort: ");
  Serial.println(currentPortNumber);


  /* ######################### Initialise all IMUs ######################### */
  // Create set of pointers to sensor class
  myICM = new ICM_20948_I2C *[NUMBER_OF_SENSORS];
  // Assign pointers to pointers of instances of sensor class
  for (int x = 0; x < NUMBER_OF_SENSORS; x++)
  {
    myICM[x] = new ICM_20948_I2C();
  }

  // Initialise IMUs
  bool initSuccess = true;
  for (byte x = 0; x < NUMBER_OF_SENSORS; x++)
  {
    myMux.setPort(x);
    myICM[x]->begin(WIRE_PORT, AD0_VAL);
    if (myICM[x]->status != ICM_20948_Stat_Ok)
    {
      SERIAL_PORT.print("IMU ");
      SERIAL_PORT.print(x);
      SERIAL_PORT.println(" did not begin! Check wiring.");
      initSuccess = false;
    }
    else
    {
      SERIAL_PORT.print("IMU ");
      SERIAL_PORT.print(x);
      SERIAL_PORT.println(" successfully initialised.");      
    }
  }

  if (initSuccess == false)
  {
    SERIAL_PORT.print("Error in IMU initialisation. Freezing...");
    while (1)
      ;
  }

  SERIAL_PORT.println("All hardware initialisation complete.");


  /* ######################### Connect to Wi-Fi ######################### */
  SERIAL_PORT.print("Connecting to Wi-Fi with ID {");
  SERIAL_PORT.print(SSID);
  SERIAL_PORT.println("} ...");
  WiFi.begin(SSID, PASSWORD);

  while (WiFi.status() != WL_CONNECTED)
  {
    delay(1000);
    SERIAL_PORT.print("Connecting to Wi-Fi with ID {");
    SERIAL_PORT.print(SSID);
    SERIAL_PORT.println("}");
  }
  SERIAL_PORT.println("Connected to Wi-Fi");
}

void loop()
{
  int delay_time;
  
  for (byte x = 0; x < NUMBER_OF_SENSORS; x++)
  {
    myMux.setPort(x); // Tell Mux to connect to this port, and this port only

    if (myICM[x]->dataReady())
    {
      myICM[x]->getAGMT();
      SERIAL_PORT.print("IMU ");
      SERIAL_PORT.print(x);
      SERIAL_PORT.println(" Data:");
      printScaledAGMT(myICM[x]);
      delay_time = 1 / SAMPLE_RATE * 1000;
    }
    else
    {
      SERIAL_PORT.print("IMU ");
      SERIAL_PORT.print(x);
      SERIAL_PORT.println(": Waiting for data");
      delay_time = 5000;
    }
  }
  delay(delay_time);
}
