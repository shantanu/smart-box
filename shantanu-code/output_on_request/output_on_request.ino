int analogPin = A0;
int data = 0;
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
      data = analogRead(analogPin);
      Serial.println(data);
    }
  }
}
