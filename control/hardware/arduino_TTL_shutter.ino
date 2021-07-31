/* Use Arduino to activate shutter
 */

int pin_TTL_out1 = 13;
char COM_in;

void setup() {
  pinMode(pin_TTL_out1, OUTPUT);     // define output pin
  digitalWrite(pin_TTL_out1, LOW);
  Serial.begin(115200);             // establish serial communication
}

void loop() { 
  if (Serial.available() > 0) {
    digitalWrite(pin_TTL_out1,LOW);
    COM_in = Serial.read();
    if(COM_in == 'o'){
      digitalWrite(pin_TTL_out1, HIGH);
      Serial.println('o');        
    }
    else if(COM_in == 'c'){
      digitalWrite(pin_TTL_out1, LOW);
      Serial.println('c');
    }
    else{
      //Serial.println('b');
    }
  }
}
