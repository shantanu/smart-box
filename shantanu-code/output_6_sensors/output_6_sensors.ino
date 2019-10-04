int photo1Pin = A0;
int photo2Pin = A1;
int joystickXPin = A2;
int joystickYPin = A3;
int soundPin = A4;
int tempPin = A5;


int photo1data = 0;
int photo2data = 0;
int joystickXdata = 0;
int joystickYdata = 0;
int sounddata = 0;
int tempdata = 0;


char userInput;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0){
    userInput = Serial.read();

    if (userInput == 'r') {
      photo1data = analogRead(photo1Pin);
      photo2data = analogRead(photo2Pin);
      
      joystickXdata = analogRead(joystickXPin);
      joystickYdata = analogRead(joystickYPin);
     
      sounddata = analogRead(soundPin);
      tempdata = analogRead(tempPin);
      

      Serial.print(photo1data);
      Serial.print(",");
      Serial.print(photo2data);
      Serial.print(",");
      Serial.print(joystickXdata);
      Serial.print(",");
      Serial.print(joystickYdata);
      Serial.print(",");
      Serial.print(sounddata);
      Serial.print(",");
      Serial.print(tempdata);
      Serial.println();
      
    }
  }
}
