#include <Wire.h>
#include <SPI.h>
#include <TimerOne.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include "Adafruit_TCS34725.h"



#define BME_SCK 13
#define BME_MISO 12
#define BME_MOSI 11
#define BME_CS 10

#define SEALEVELPRESSURE_HPA (1013.25)

Adafruit_BME280 bme; // I2C

Adafruit_BNO055 bno = Adafruit_BNO055(55);

/* Initialise with specific int time and gain values */
Adafruit_TCS34725 tcs = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_700MS, TCS34725_GAIN_1X);

int PIRPin = 2;
int val = 0;

int micPin = A0;
unsigned int sample;

unsigned long delayTime;

volatile boolean collectFlag = false;

char userInput;


void setup() {
    Serial.begin(115200);
    while(!Serial);    // time to get serial running    

    
    //Serial.println(F("BME280 test"));

    unsigned status;
    
    // default settings
    status = bme.begin();  
    // You can also pass in a Wire library object like &Wire2
    // status = bme.begin(0x76, &Wire2)
    if (!status) {
        Serial.println("Could not find a valid BME280 sensor, check wiring, address, sensor ID!");
        Serial.print("SensorID was: 0x"); Serial.println(bme.sensorID(),16);
        Serial.print("        ID of 0xFF probably means a bad address, a BMP 180 or BMP 085\n");
        Serial.print("   ID of 0x56-0x58 represents a BMP 280,\n");
        Serial.print("        ID of 0x60 represents a BME 280.\n");
        Serial.print("        ID of 0x61 represents a BME 680.\n");
        while (1) delay(10);
    }
    
    //Serial.println("-- Default Test --");
    delayTime = 100;

    //Serial.println("Orientation Sensor Test"); Serial.println("");
  
    /* Initialise the sensor */
    if(!bno.begin())
    {
      /* There was a problem detecting the BNO055 ... check your connections */
      Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
      while(1);
    }

    //Serial.println("Color Sensor");
    if (tcs.begin()) {
      //Serial.println("Found sensor");
    } else {
      Serial.println("No TCS34725 found ... check your connections");
      while (1);
    }
    
    delay(1000);
      
    bno.setExtCrystalUse(true);
    pinMode(PIRPin, INPUT);     // declare sensor as input



    //Serial.println("Starting Sensor Readings");
}

void loop(){
  
  
  if (Serial.available() > 0) {
    userInput = Serial.read();

    if (userInput == 'r'){
      printSensorValues();
    }
  }
  
}


// We define a 18 length array for the sensor readings
// [PIR, MIC, Color Temp, Lumosity, R, G, B, C, Temperature, Pressure, Approx Altitude, Humidity,
// Accel X (m/s), Accel Y, Accel Z, Magnet X(uT), Magnet Y, Magnet Z]

int PIR = 0;
int MIC = 0;
double COLOR[6] = {0};
double BME[4] = {0};
double BNO[6] = {0};

double READINGS[18] = {0};
void printSensorValues(){
  //Serial.println("hi");
  unsigned long startTime = millis();
  // 1. PIR
  READINGS[0] = (digitalRead(PIRPin) == HIGH) ? 1 : 0;
  //Serial.print("PIR");

  // 2. Audio
  READINGS[1] = getMicValue(10);
  //Serial.print("MIC");

  // 3. TCS34725 Color Sensor
  // Channels: [Color Temp, Lumosity, R, G, B, C]
  //double COLOR[6] = {0};
  getColorValues(READINGS + 2, 6);
  //printArray(COLOR, 6);

  // 4. BME280 Temp/Baro/Hum
  // Channels: [Temperature, Pressure, Approx Altitude, Humidity]
  //double BME[4] = {0.0};
  getBMEValues(READINGS + 8, 4);
  //printArray(BME, 4);

  // 5. BNO055 Accel/Magnet
  // Channels: [Accel X (m/s), Accel Y, Accel Z, Magnet X(uT), Magnet Y, Magnet Z]
  //double BNO[6] = {0};
  getBNOValues(READINGS + 12, 6);
  //printArray(BNO, 6);

  printArray(READINGS, 18);
    
}

double getMicValue(int sampleWindow) {
   unsigned long startMillis= millis();  // Start of sample window
   unsigned int peakToPeak = 0;   // peak-to-peak level

   unsigned int signalMax = 0;
   unsigned int signalMin = 1024;

   // collect data for 50 mS
   while (millis() - startMillis < sampleWindow)
   {
      sample = analogRead(0);
      if (sample < 1024)  // toss out spurious readings
      {
         if (sample > signalMax)
         {
            signalMax = sample;  // save just the max levels
         }
         else if (sample < signalMin)
         {
            signalMin = sample;  // save just the min levels
         }
      }
   }
   peakToPeak = signalMax - signalMin;  // max - min = peak-peak amplitude
   double volts = (peakToPeak * 5.0) / 1024;  // convert to volts

    return volts;
}

void getColorValues(double* values, int len){
  uint16_t r, g, b, c, colorTemp, lux;

  tcs.getRawData(&r, &g, &b, &c);
  // colorTemp = tcs.calculateColorTemperature(r, g, b);
  colorTemp = tcs.calculateColorTemperature_dn40(r, g, b, c);
  lux = tcs.calculateLux(r, g, b);

  values[0] = colorTemp;
  values[1] = lux;
  values[2] = r;
  values[3] = g;
  values[4] = b;
  values[5] = c;
}

void getBMEValues(double* values, int len){
  values[0] = bme.readTemperature();
  values[1] = bme.readPressure() / 100.0F;
  values[2] = bme.readAltitude(SEALEVELPRESSURE_HPA);
  values[3] = bme.readHumidity();
}

void getBNOValues(double* values, int len){
  /* Get a new sensor event */ 
  sensors_event_t event; 
  bno.getEvent(&event);
  

  imu::Vector<3> accel = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
  values[0] = accel.x();
  values[1] = accel.y();
  values[2] = accel.z();
  
  imu::Vector<3> magnet = bno.getVector(Adafruit_BNO055::VECTOR_MAGNETOMETER);

   values[3] = magnet.x();
   values[4] = magnet.y();
   values[5] = magnet.z();
 
}


void printArray(double* arr, int len){
  for (int i = 0; i < len-1; i ++){
    Serial.print(arr[i], 3); Serial.print(F(", "));
  }
  Serial.print(arr[len-1]);
  Serial.println();
}
