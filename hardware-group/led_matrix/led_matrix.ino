#include "Arduino_LED_Matrix.h"

ArduinoLEDMatrix matrix;

const uint32_t start[] = 
{
    0xFFFFFFFF,
    0xFFFFFFFF,
    0xFFFFFFFF
};

void setup()
{
    Serial.begin(115200);
    matrix.begin();
}

void loop()
{
    matrix.loadFrame(start);
    delay(30);
}