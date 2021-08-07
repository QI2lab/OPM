/* Use Arduino to activate shutter
 */

int pin_TTL_out1 = 32;
char COM_in;
char state;

void setup() {
  pinMode(pin_TTL_out1, OUTPUT);     // define output pin
  digitalWrite(pin_TTL_out1, LOW);
  Serial.begin(115200);             // establish serial communication
  state = 'c';
}

void loop() { 
  if (Serial.available() > 0) {
    COM_in = Serial.read();
    if(COM_in == 'o'){
      digitalWrite(pin_TTL_out1, HIGH);
      state = 'o';
      Serial.println(state);
    }
    else if(COM_in == 'c'){
      digitalWrite(pin_TTL_out1, LOW);
      state = 'c';
      Serial.print(state);
    }
    else if(COM_in == 's') {
      Serial.println(state);
    }
  }
}
