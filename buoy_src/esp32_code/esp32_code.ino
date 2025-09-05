#include <M5StickCPlus.h>
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiMulti.h>
#include <HTTPClient.h>
#include <driver/i2s.h>

WiFiMulti wifiMulti;
HTTPClient http;

const char *wifi_ssid = "Verizon_T6YMSZ";
const char *server_ip = "http://192.168.1.85:5000/";

const String dev_name = "moby_buoy_1";
const String location = "39.042388, -77.550108";

bool is_registered = false;

#define PIN_CLK     0
#define PIN_DATA    34
#define SAMPLE_RATE 6000
#define DURATION_SEC 1
#define CHANNELS    1

#define TOTAL_SAMPLES (SAMPLE_RATE * DURATION_SEC)
#define READ_LEN      512

int16_t rawBuffer[READ_LEN / 2];
float* audioBuffer;
int audioIndex = 0;

char *postData;

void reset() {
  delay(5000);
  // Optionally clear the LCD or show error
}

void i2sInit() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ALL_RIGHT,
#if ESP_IDF_VERSION > ESP_IDF_VERSION_VAL(4, 1, 0)
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
#else
        .communication_format = I2S_COMM_FORMAT_I2S,
#endif
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 2,
        .dma_buf_len   = 128,
    };

    i2s_pin_config_t pin_config;
#if (ESP_IDF_VERSION > ESP_IDF_VERSION_VAL(4, 3, 0))
    pin_config.mck_io_num = I2S_PIN_NO_CHANGE;
#endif
    pin_config.bck_io_num = I2S_PIN_NO_CHANGE;
    pin_config.ws_io_num = PIN_CLK;
    pin_config.data_out_num = I2S_PIN_NO_CHANGE;
    pin_config.data_in_num = PIN_DATA;

    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_config);
    i2s_set_clk(I2S_NUM_0, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_16BIT, I2S_CHANNEL_MONO);
}


void API_send_task(void * arg) {
  while (1) {
    if(audioIndex < TOTAL_SAMPLES) {
      vTaskDelay(100 / portTICK_PERIOD_MS);
      continue;
    }

    String callpoint = server_ip;
    callpoint += "update";

    Serial.println("[HTTP] update begin...\n");
    M5.Lcd.println("[HTTP] update begin...");

    http.begin(callpoint);
    http.addHeader("Content-Type", "application/json");

    String data = "[";
    for (int i = 0; i < TOTAL_SAMPLES; i++) {
      data += String(audioBuffer[i]);
      if (i < TOTAL_SAMPLES - 1) data += ", ";
      if (i % 500 == 0) yield();
    }
    data += "]";

    sprintf(postData, "{\"name\":\"%s\", \"data\":%s}", dev_name.c_str(), data.c_str());

    int httpCode = http.POST(postData);
    if (httpCode == 200) {
      Serial.printf("POST audio OK, code: %d\n", httpCode);
      M5.Lcd.printf("POST audio OK: %d\n", httpCode);
    } else {
      Serial.printf("POST audio FAIL: %d\n", httpCode);
      M5.Lcd.printf("POST audio FAIL: %d\n", httpCode);
    }
    http.end();

    audioIndex = 0;
  }
}

void API_register() {
  String callpoint = server_ip;
  callpoint += "register";

  Serial.println("[HTTP] register begin...");
  M5.Lcd.println("[HTTP] register begin...");

  http.begin(callpoint);
  http.addHeader("Content-Type", "application/json");

  String postData = "{\"name\":\"" + dev_name + "\", \"location\":\"" + location + "\"}";

  Serial.println("[HTTP] register POST...");
  M5.Lcd.println("[HTTP] register POST...");

  int httpCode = http.POST(postData);

  if (httpCode != 200) {
    Serial.printf("POST register failed: %s\n", http.errorToString(httpCode).c_str());
    M5.Lcd.printf("Reg FAIL: %s\n", http.errorToString(httpCode).c_str());
  } else {
    Serial.printf("POST register OK: code %d\n", httpCode);
    M5.Lcd.printf("Reg OK: %d\n", httpCode);
    is_registered = true;
    String response = http.getString();
  }

  http.end();
}

void setup() {
  M5.begin();
  M5.Lcd.setRotation(3);
  M5.Lcd.fillScreen(BLACK);
  M5.Lcd.setTextSize(1);
  M5.Lcd.setCursor(0, 0);

  Serial.begin(115200);
  delay(1000);

  i2sInit();

  Serial.println("\nConnecting Wifi...\n");
  M5.Lcd.println("Connecting Wifi...");

  wifiMulti.addAP(wifi_ssid, wifi_passwd);

  audioBuffer = (float*)malloc(TOTAL_SAMPLES * sizeof(float));
  postData = (char*)malloc(60000);

  Serial.printf("Audio: 0x%X, Data: 0x%X\n", audioBuffer, postData);
  M5.Lcd.printf("Audio: 0x%X\nData: 0x%X\n", audioBuffer, postData);

  audioIndex = 0;
  xTaskCreate(API_send_task, "API_send_task", 16384, NULL, 1, NULL);
}
int cnt = 0;

void loop() {
  M5.update();

  if (M5.BtnA.wasPressed()) {
    Serial.println("\nRestarting...");
    M5.Lcd.println("Restarting...");
    delay(1000);
    ESP.restart();
  }

  if((wifiMulti.run() != WL_CONNECTED)) {
    Serial.println("connect failed");
    M5.Lcd.println("WiFi Failed");
    reset();
    return;
  }

  if(!is_registered) {
    API_register();
    delay(50);
  }
  else if(audioIndex == 0) {
    cnt++;
    if(cnt % 3 == 0)
    {     
      M5.Lcd.fillScreen(BLACK);
      M5.Lcd.setCursor(0, 0);
    }

    Serial.println("Starting Recording");
    M5.Lcd.println("Recording...");

    size_t bytesread;

    while (audioIndex < TOTAL_SAMPLES) {
        i2s_read(I2S_NUM_0, (char *)rawBuffer, READ_LEN, &bytesread, portMAX_DELAY);
        int samples_read = bytesread / sizeof(int16_t);
        for (int i = 0; i < samples_read && audioIndex < TOTAL_SAMPLES; ++i) {
            float normalized = rawBuffer[i] / 32768.0f;
            audioBuffer[audioIndex++] = normalized;
        }
    }
    Serial.println("Recording Over");
    M5.Lcd.println("Recording Done");
  }
}
