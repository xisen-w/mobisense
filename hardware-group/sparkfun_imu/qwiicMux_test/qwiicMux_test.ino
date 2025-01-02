#include <Wire.h>

#define MUX_ADDR 0x70 // Default Qwiic Mux Shield address

void setup()
{
  Serial.begin(115200);
  Serial.println("Starting Qwiic Mux Shield Connection Test...");
  Wire1.begin();

  // Check if Mux Shield is connected
  bool initialised = false;
  int MAX_ITER = 10;
  int iter = 0;
  Wire1.beginTransmission(MUX_ADDR);
  while(!initialised && iter <= MAX_ITER)
  {
    if (Wire1.endTransmission() == 0)
    {
      Serial.println("Mux Shield detected at address 0x70.");
      initialised = true;
    }
    else
    {
      Serial.println("Mux Shield not detected. Please check wiring.");
      delay(1000);
      iter++;
    }
  }

  Serial.println("Mux Shield connection test complete.");
}

void loop()
{
  // no loop.
}