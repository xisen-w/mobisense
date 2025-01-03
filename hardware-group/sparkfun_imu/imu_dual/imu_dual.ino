#include <Wire.h>
#include <ICM_20948.h>
#include <SparkFun_I2C_Mux_Arduino_Library.h>

#define SERIAL_PORT Serial
#define WIRE_PORT Wire1 // Qwiic I2C communication

#define AD0_VAL 1 // The value of the last bit of the IMU I2C address. Defaults to 1.
#define NUMBER_OF_SENSORS 2
#define I2C_CLOCK_SPEED 400000
#define MUX_ADDR 0x70

QWIICMUX myMux;
ICM_20948_I2C **myICM; // Create pointer to a set of pointers to the sensor class

void setup()
{
  SERIAL_PORT.begin(115200);
  while (!SERIAL_PORT)
  {
    // hang
  };
  SERIAL_PORT.println("Initialising IMUs via Qwiic I2C MUX...");

  // Open Wire1() for Qwiix I2C
  WIRE_PORT.begin();
  WIRE_PORT.setClock(I2C_CLOCK_SPEED); // Set I2C clock speed

  // Create set of pointers to sensor class
  myICM = new ICM_20948_I2C *[NUMBER_OF_SENSORS];
  // Assign pointers to pointers of instances of sensor class
  for (int x = 0; x < NUMBER_OF_SENSORS; x++)
  {
    myICM[x] = new ICM_20948_I2C();
  }

  // Check MUX connection
  if (myMux.begin(MUX_ADDR, WIRE_PORT) == false)
  {
    SERIAL_PORT.println("Mux not detected. Reset Uno Board to re-detect. Freezing...");
    while(1);
  }
  SERIAL_PORT.println("Mux detected. Initialising IMUs...");

  byte currentPortNumber = myMux.getPort();
  Serial.print("CurrentPort: ");
  Serial.println(currentPortNumber);

  // Initialise all IMUs
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

  SERIAL_PORT.println("All Initialisation complete.");
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
      delay_time = 500;
    }
    else
    {
      SERIAL_PORT.print("IMU ");
      SERIAL_PORT.print(x);
      SERIAL_PORT.println(": Waiting for data");
      delay_time = 500;
    }
  }
  delay(delay_time);
}
// void loop()
// {
//   if (myICM.dataReady())
//   {
//     myICM.getAGMT(); // The values are only updated when you call 'getAGMT'
//     printScaledAGMT(&myICM); // Print scaled sensor data
//     delay(30);
//   }
//   else
//   {
//     SERIAL_PORT.println("Waiting for data");
//     delay(500);
//   }
// }

// Below here are some helper functions to print the data nicely!

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

void printRawAGMT(ICM_20948_AGMT_t agmt)
{
  SERIAL_PORT.print("RAW. Acc [ ");
  printPaddedInt16b(agmt.acc.axes.x);
  SERIAL_PORT.print(", ");
  printPaddedInt16b(agmt.acc.axes.y);
  SERIAL_PORT.print(", ");
  printPaddedInt16b(agmt.acc.axes.z);
  SERIAL_PORT.print(" ], Gyr [ ");
  printPaddedInt16b(agmt.gyr.axes.x);
  SERIAL_PORT.print(", ");
  printPaddedInt16b(agmt.gyr.axes.y);
  SERIAL_PORT.print(", ");
  printPaddedInt16b(agmt.gyr.axes.z);
  SERIAL_PORT.print(" ], Mag [ ");
  printPaddedInt16b(agmt.mag.axes.x);
  SERIAL_PORT.print(", ");
  printPaddedInt16b(agmt.mag.axes.y);
  SERIAL_PORT.print(", ");
  printPaddedInt16b(agmt.mag.axes.z);
  SERIAL_PORT.print(" ], Tmp [ ");
  printPaddedInt16b(agmt.tmp.val);
  SERIAL_PORT.print(" ]");
  SERIAL_PORT.println();
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

void printScaledAGMT(ICM_20948_I2C *sensor)
{
  SERIAL_PORT.print("Scaled. Acc (mg) [ ");
  printFormattedFloat(sensor->accX(), 5, 2);
  SERIAL_PORT.print(", ");
  printFormattedFloat(sensor->accY(), 5, 2);
  SERIAL_PORT.print(", ");
  printFormattedFloat(sensor->accZ(), 5, 2);
  SERIAL_PORT.print(" ], Gyr (DPS) [ ");
  printFormattedFloat(sensor->gyrX(), 5, 2);
  SERIAL_PORT.print(", ");
  printFormattedFloat(sensor->gyrY(), 5, 2);
  SERIAL_PORT.print(", ");
  printFormattedFloat(sensor->gyrZ(), 5, 2);
  SERIAL_PORT.print(" ], Mag (uT) [ ");
  printFormattedFloat(sensor->magX(), 5, 2);
  SERIAL_PORT.print(", ");
  printFormattedFloat(sensor->magY(), 5, 2);
  SERIAL_PORT.print(", ");
  printFormattedFloat(sensor->magZ(), 5, 2);
  SERIAL_PORT.print(" ], Tmp (C) [ ");
  printFormattedFloat(sensor->temp(), 5, 2);
  SERIAL_PORT.print(" ]");
  SERIAL_PORT.println();
}