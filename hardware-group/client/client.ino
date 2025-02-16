#define USE_UDP   // toggle for http/udp, comment off to use HTTP
#define DEBUG     // toggle for debug: displaying sending freq in Hz

#include <Wire.h>
#include <WiFiS3.h>                             // For Wi-Fi connectivity
#ifdef USE_UDP
  #include <WiFiUdp.h>                          // For UDP requests
#else
  #include <ArduinoHttpClient.h>                // For HTTP requests
#endif
#include <ICM_20948.h>                          // For IMUs
#include <SparkFun_I2C_Mux_Arduino_Library.h>   // For MUX
#include <ArduinoGraphics.h>
#include "Arduino_LED_Matrix.h"

#include "Constants.h"                          // Pre-defined constants

WiFiClient WIFI;                                // Wi-Fi client
#ifdef USE_UDP
  WiFiUDP udp;                                  // UDP Client
#else
  HttpClient client(WIFI, SERVER, PORT);        // HTTP client
#endif
QWIICMUX myMux;                                 // Create instance of the Qwiic Mux class
ICM_20948_I2C **myICM;                          // Create pointer to a set of pointers to the sensor class
ArduinoLEDMatrix matrix;                        // LED matrix

// global variables
char payload[256];            // initialise fixed-sized json payload
int countDown = 300 * 18;          // delay before start mark
bool startRecording = false;  // Flag to check if start recording

// Debug parameters
#ifdef DEBUG
  unsigned long lastSendTime = 0;
  float sendingFrequency = 0.0;   // Frequency in Hz
  int packetCount = 0;            // Count packets sent
  unsigned long lastLoopTime = 0;
  const int targetInterval = 10;  // 100 Hz = 10 ms interval
#endif

void setup()
{
  SERIAL_PORT.begin(115200);
  // while (!SERIAL_PORT)
  // {
  //   // hang
  // };


  /* ######################### Initialising LED Matrix ######################### */
  matrix.begin();
  SERIAL_PORT.println("LED Matrix Initialised.");


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

    // Initialise DMP module for the set IMU
    bool success = true; // Use success to show if the DMP configuration was successful
    success &= (myICM[x]->initializeDMP() == ICM_20948_Stat_Ok); 
    success &= (myICM[x]->enableDMPSensor(INV_ICM20948_SENSOR_GAME_ROTATION_VECTOR) == ICM_20948_Stat_Ok); 
    success &= (myICM[x]->setDMPODRrate(DMP_ODR_Reg_Quat6, 0) == ICM_20948_Stat_Ok); 

    // Enable the FIFO
    success &= (myICM[x]->enableFIFO() == ICM_20948_Stat_Ok);

    // Enable the DMP
    success &= (myICM[x]->enableDMP() == ICM_20948_Stat_Ok);

    // Reset DMP
    success &= (myICM[x]->resetDMP() == ICM_20948_Stat_Ok);

    // Reset FIFO
    success &= (myICM[x]->resetFIFO() == ICM_20948_Stat_Ok);  

    if (success)  // DMP module check
    {
      SERIAL_PORT.println(F("DMP enabled!"));
    } 
    else 
    {
      SERIAL_PORT.println(F("Enable DMP failed!"));
      SERIAL_PORT.println(F("Please check that you have uncommented line 29 (#define ICM_20948_USE_DMP) in ICM_20948_C.h..."));
      while (1)
        ;
    }
  }

  if (initSuccess == false)  // Overall IMU initialisation check
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
    WiFi.begin(SSID, PASSWORD);
    SERIAL_PORT.print("Connecting to Wi-Fi with ID {");
    SERIAL_PORT.print(SSID);
    SERIAL_PORT.println("} ...");
  }
  SERIAL_PORT.println("Connected to Wi-Fi");

  // Open UDP port for UDP packet sending
  udp.begin(LOCAL_UDP_PORT);
}

void loop()
{
  // Re-initialise payload buffer
  // snprintf(payload, sizeof(payload), "{\"imu_data\":[");
  snprintf(payload, sizeof(payload), "");
  // countDown before matrix lights up
  if (countDown > 0)
  {
    countDown -= 1;
  }
  Serial.print("count down: ");
  Serial.println(countDown);

  bool allDataReady = true;     // Flag to check if all IMUs have data
  for (byte x = 0; x < NUMBER_OF_SENSORS; x++)
  {
    myMux.setPort(x); // Tell MUX to connect to this port, and this port only

    // DMP
    icm_20948_DMP_data_t data;
    myICM[x]->readDMPdataFromFIFO(&data);
    // dubugging
    // SERIAL_PORT.println(myICM[x]->dataReady());
    // SERIAL_PORT.println(myICM[x]->status);

    if (myICM[x]->dataReady() || (myICM[x]->status == ICM_20948_Stat_FIFOMoreDataAvail))
    {
      myICM[x]->getAGMT(); // Retrieve data for this IMU

      double roll = 0.0, pitch = 0.0, yaw = 0.0; // Default values

      if ((data.header & DMP_header_bitmap_Quat6) > 0) // We have asked for GRV data so we should receive Quat6
      {
          // Scale to +/- 1
          double q1 = ((double)data.Quat6.Data.Q1) / 1073741824.0; // Convert to double. Divide by 2^30
          double q2 = ((double)data.Quat6.Data.Q2) / 1073741824.0; // Convert to double. Divide by 2^30
          double q3 = ((double)data.Quat6.Data.Q3) / 1073741824.0; // Convert to double. Divide by 2^30

          double q0 = sqrt(1.0 - ((q1 * q1) + (q2 * q2) + (q3 * q3)));

          double qw = q0; 
          double qx = q2;
          double qy = q1;
          double qz = -q3;

          // roll (x-axis rotation)
          double t0 = +2.0 * (qw * qx + qy * qz);
          double t1 = +1.0 - 2.0 * (qx * qx + qy * qy);
          roll = atan2(t0, t1) * 180.0 / PI;

          // pitch (y-axis rotation)
          double t2 = +2.0 * (qw * qy - qx * qz);
          t2 = t2 > 1.0 ? 1.0 : t2;
          t2 = t2 < -1.0 ? -1.0 : t2;
          pitch = asin(t2) * 180.0 / PI;

          // yaw (z-axis rotation)
          double t3 = +2.0 * (qw * qz + qx * qy);
          double t4 = +1.0 - 2.0 * (qy * qy + qz * qz);
          yaw = atan2(t3, t4) * 180.0 / PI;
      }

      // Create JSON payload for this IMU
      char imuData[128];
      snprintf(imuData, sizeof(imuData),
        "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f%s",
        myICM[x]->accX(),myICM[x]->accY(),myICM[x]->accZ(),
        myICM[x]->gyrX(),myICM[x]->gyrY(),myICM[x]->gyrZ(),
        roll, pitch, yaw, 
        (x < NUMBER_OF_SENSORS - 1) ? "," : ""  // Add comma except for last IMU
      );
      // Append IMU data to payload
      strncat(payload, imuData, sizeof(payload) - strlen(payload) - 1);
      // Serial.println(payload);
      // Serial.print("Size needed: ");
      // Serial.println(strlen(payload));
    }
    else
    {
      SERIAL_PORT.print("IMU ");
      SERIAL_PORT.print(x);
      SERIAL_PORT.println(": Data not ready. Skipping...");
      allDataReady = false; // Mark as incomplete if any IMU data is missing
    }
  }

  // Check if all data is ready
  if (allDataReady)
  {
    // SERIAL_PORT.println("All IMU readings obtained. Sending data...");
    // SERIAL_PORT.println(payload); // Print payload for debugging

    // Send JSON payload to the server
    #ifdef USE_UDP
      // UDP Transmission
      // Debug
      #ifdef DEBUG
        unsigned long currentTime = millis();  // Get current time in milliseconds
        // Compute sending frequency (only after first packet)
        if (lastSendTime > 0) 
        {
          unsigned long elapsedTime = currentTime - lastSendTime;
          sendingFrequency = 1000.0 / elapsedTime;  // Convert ms to Hz
        }
        lastSendTime = currentTime;  // Update last send time
        packetCount++;
        // SERIAL_PORT.print("sending freq: ");
        SERIAL_PORT.println(sendingFrequency);
      #endif

      // recording start event
      if (countDown > 0 && !startRecording)
      {
        snprintf(payload, sizeof(payload), "starting...");
        udp.beginPacket(SERVER, PORT);
        udp.write((uint8_t*)payload, strlen(payload));
        udp.endPacket();
        matrix.textFont(Font_4x6);
        matrix.beginText(0, 1, 0xFFFFFF);
        matrix.println(String(countDown));
        matrix.endText();
      }
      if (countDown == 0 && !startRecording)
      {
        startRecording = true;
        snprintf(payload, sizeof(payload), "start");
        udp.beginPacket(SERVER, PORT);
        udp.write((uint8_t*)payload, strlen(payload));
        udp.endPacket();
        // lights matrix
        matrix.loadFrame(startFrame);
      }
      if (startRecording)
      {
        // Serial.println(payload);
        udp.beginPacket(SERVER, PORT);
        udp.write((uint8_t*)payload, strlen(payload));
        udp.endPacket();
      }
    #else
      // HTTP Transmission
      client.beginRequest();
      client.post(API_PATH);
      client.sendHeader("Content-Type", "application/json");
      client.sendHeader("Content-Length", payload.length());
      client.beginBody();
      client.print(payload);
      client.endRequest();

      // Check response
      int statusCode = client.responseStatusCode();
      String response = client.responseBody();

      SERIAL_PORT.print("Status Code: ");
      SERIAL_PORT.println(statusCode);
      SERIAL_PORT.print("Response: ");
      SERIAL_PORT.println(response);
    #endif
  }
  else
  {
    SERIAL_PORT.println("Error: Not all IMU data is ready. No data sent.");
    // lights up LED
    matrix.loadFrame(dangerFrame);
  }
}
