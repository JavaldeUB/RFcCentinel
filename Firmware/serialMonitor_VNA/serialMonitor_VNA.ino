
#include <Arduino.h>
#include "wiring_private.h"
#include <Scheduler.h>

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
  pinMode(RED, OUTPUT);
  pinMode(GREEN, OUTPUT);
  pinMode(LED_PWR, OUTPUT);
  digitalWrite(RED, HIGH); // turn the LED off by making the voltage LOW
  digitalWrite(GREEN, HIGH); // turn the LED off by making the voltage LOW
  digitalWrite(LED_PWR,LOW); // turn the LED off by making the voltage LOW
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
      digitalWrite(RED,LOW);
      delay(100);
    }  
    Serial.println("");
    Serial.print("Sent Command: ");
    Serial.println(command);
  }
  else{
    digitalWrite(RED,HIGH);
    if(Serial1.available() > 0) {
      digitalWrite(GREEN,LOW);
      char receivedChar = Serial1.read();
      Serial.print(receivedChar);
    }
    else{
      digitalWrite(GREEN,HIGH);
    }    
  }
  //yield();
}

void loop2(){
  if(Serial1.available() > 0) {
    digitalWrite(GREEN,LOW);
    char receivedChar = Serial1.read();
    Serial.print(receivedChar);
  }
  else{
    digitalWrite(GREEN,HIGH);
  }
}