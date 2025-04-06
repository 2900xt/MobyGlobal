void setup() {
  Serial.begin(115200);
  #ifdef M5STACK
    Serial.println("M5Stack Core detected");
  #endif
  #ifdef M5STICK_C
    Serial.println("M5StickC detected");
  #endif
  #ifdef M5CORE2
    Serial.println("M5Core2 detected");
  #endif
  #ifdef M5ATOM
    Serial.println("M5Atom detected");
  #endif
  Serial.
}

void loop() {}
