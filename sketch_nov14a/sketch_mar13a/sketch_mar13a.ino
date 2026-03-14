// Advanced EMG Signal Processing
// Works with Arduino Serial Plotter

const int EMG_PIN = A0;

int rawSignal = 0;
int envelope = 0;

const int threshold = 520;   // adjust based on your sensor

void setup() {
  Serial.begin(115200);
}

void loop() {

  rawSignal = analogRead(EMG_PIN);

  // Rectification (convert negative-like signals to positive)
  int rectified = abs(rawSignal - 512);

  // Envelope detection (smooth signal)
  envelope = (envelope * 0.95) + (rectified * 0.05);

  // Send values to Serial Plotter
  Serial.print(rawSignal);
  Serial.print(",");
  Serial.print(envelope);
  Serial.print(",");
  Serial.println(threshold);

  delay(2);  // fast sampling for smooth waveform
}