# 1 "c:\\Users\\javit\\Documents\\curso23-24\\Investigacion\\EHU\\Firmware\\serialMonitor_VNA\\serialMonitor_VNA.ino"

# 3 "c:\\Users\\javit\\Documents\\curso23-24\\Investigacion\\EHU\\Firmware\\serialMonitor_VNA\\serialMonitor_VNA.ino" 2
# 4 "c:\\Users\\javit\\Documents\\curso23-24\\Investigacion\\EHU\\Firmware\\serialMonitor_VNA\\serialMonitor_VNA.ino" 2
# 5 "c:\\Users\\javit\\Documents\\curso23-24\\Investigacion\\EHU\\Firmware\\serialMonitor_VNA\\serialMonitor_VNA.ino" 2

#define RED 22
#define BLUE 24
#define GREEN 23
#define LED_PWR 25

//UART serialVNA(digitalPinToPinName(4), digitalPinToPinName(3), NC, NC);


void setup() {
  // Start Serial for communication with the Arduino IDE Serial Monitor
  Serial.begin(115200);
  // Start Serial1 with a baud rate of your choice (e.g., 9600)
  Serial1.begin(115200,SERIAL_8N1);
  // initialize the digital Pin as an output
  pinMode(22, OUTPUT);
  pinMode(23, OUTPUT);
  pinMode(25, OUTPUT);
  digitalWrite(22, HIGH); // turn the LED off by making the voltage LOW
  digitalWrite(23, HIGH); // turn the LED off by making the voltage LOW
  digitalWrite(25,LOW); // turn the LED off by making the voltage LOW
  //Second loop for listenning the answer of the serialVNA
  //Scheduler.startLoop(loop2);
}

void loop() {
  // Check if there is data available on Serial (USB)
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\r');
    command = command +'\r';
    for(int i=0; i<command.length();i++){
      char currentC = command.charAt(i);
      Serial1.write(currentC);
      Serial.print(byte(currentC));
      digitalWrite(22,LOW);
      delay(100);
    }
    Serial.println("");
    Serial.print("Sent Command: ");
    Serial.println(command);
  }
  else{
    digitalWrite(22,HIGH);
    if(Serial1.available() > 0) {
      digitalWrite(23,LOW);
      char receivedChar = Serial1.read();
      Serial.print(receivedChar);
    }
    else{
      digitalWrite(23,HIGH);
    }
  }
  //yield();
}

void loop2(){
  if(Serial1.available() > 0) {
    digitalWrite(23,LOW);
    char receivedChar = Serial1.read();
    Serial.print(receivedChar);
  }
  else{
    digitalWrite(23,HIGH);
  }
}
