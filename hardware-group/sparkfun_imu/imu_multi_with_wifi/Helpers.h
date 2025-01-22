#ifndef HELPERS_H
#define HELPERS_H

#include <ICM_20948.h> // Required for the ICM_20948 sensor library

void printPaddedInt16b(int16_t val);
void printRawAGMT(ICM_20948_AGMT_t agmt);
void printFormattedFloat(float val, uint8_t leading, uint8_t decimals);
void printScaledAGMT(ICM_20948_I2C *sensor);

#endif