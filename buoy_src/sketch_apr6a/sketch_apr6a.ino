#include <M5StickCPlus.h>
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiMulti.h>
#include <HTTPClient.h>
#include <driver/i2s.h>

WiFiMulti wifiMulti;
HTTPClient http;

const char *wifi_ssid = "Verizon_T6YMSZ";
const char *wifi_passwd = "ripe9-gram-tag";
const char *server_ip = "http://192.168.1.239:5000/";

const String dev_name = "moby_buoy_1";
const String location = "39.042388, -77.550108";

bool is_registered = false;

#define PIN_CLK     0
#define PIN_DATA    34
#define SAMPLE_RATE 6000
#define DURATION_SEC 1
#define CHANNELS    1

// Total number of samples to collect
#define TOTAL_SAMPLES (SAMPLE_RATE * DURATION_SEC)
#define READ_LEN      512

int16_t rawBuffer[READ_LEN / 2];
float* audioBuffer;
int audioIndex = 0;

char *postData;

void reset() {
  delay(5000);
  //M5.Lcd.fillRect(0, 0, 160, 80, BLACK); 
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
    if(audioIndex < TOTAL_SAMPLES) 
    {
      vTaskDelay(100 / portTICK_PERIOD_MS);
      continue;
    }

    String callpoint = server_ip;
    callpoint += "update";

    Serial.println("[HTTP] update begin...\n");

    http.begin(callpoint);  // Point to your server
    http.addHeader("Content-Type", "application/json");  // Set the content type
    String data = "[";
    for (int i = 0; i < TOTAL_SAMPLES; i++) {
      data += String(audioBuffer[i]);
      if (i < TOTAL_SAMPLES - 1) data += ", ";  
      if (i % 500 == 0) yield();  // Let watchdog breathe
    }
    data += "]";

    sprintf(postData, "{\"name\":\"%s\", \"data\":%s}", dev_name.c_str(), data.c_str());

    int httpCode = http.POST(postData);
    if (httpCode == 200) {
      Serial.printf("POST audio OK, code: %d\n", httpCode);
    } else {
      Serial.printf("POST audio FAIL: %d\n", httpCode);
    }
    http.end();

    audioIndex = 0;
  }
}

void API_register() {
  String callpoint = server_ip;
  callpoint += "register";

  Serial.println("[HTTP] register begin...");
  M5.Lcd.printf("[HTTP] register begin...\n");
  http.begin(callpoint);  // Point to your server

  http.addHeader("Content-Type", "application/json");  // Set the content type

  String postData = "{\"name\":\"" + dev_name + "\", \"location\":\"" + location + "\"}";

  Serial.println("[HTTP] register POST...");
  M5.Lcd.printf("[HTTP] register POST...\n");
  int httpCode = http.POST(postData);  // Make the POST request

  if (httpCode != 200) {
    Serial.printf("POST register failed: %s\n", http.errorToString(httpCode).c_str());
  } else {
    Serial.printf("POST register OK: code %d\n", httpCode);
    is_registered = true;
    String response = http.getString();
    //Serial.println("Response: " + response);
  }

  http.end();
}

void setup() {
  M5.begin();
  Serial.begin(115200);
  delay(1000);

  i2sInit();

  Serial.println("\nConnecting Wifi...\n");
  wifiMulti.addAP(wifi_ssid, wifi_passwd);

  audioBuffer = (float*)malloc(TOTAL_SAMPLES * sizeof(float));
  postData = (char*)malloc(60000);
  Serial.printf("Audio: 0x%X, Data: 0x%X\n", audioBuffer, postData);
  M5.Lcd.printf("Audio: 0x%X, Data: 0x%X\n", audioBuffer, postData);

  audioIndex = 0;
  xTaskCreate(API_send_task, "API_send_task", 16384, NULL, 1, NULL);
}

void loop() {  
  M5.update();
  M5.Lcd.setCursor(0, 0);

  if (M5.BtnA.wasPressed()) {
    Serial.println("\nRestarting...");
    M5.Lcd.printf("\nRestarting...");
    delay(1000);  // Optional delay for user feedback
    ESP.restart();  // Restart the M5StickC
  }

  if((wifiMulti.run() != WL_CONNECTED)) {
    Serial.println("connect failed");
    M5.Lcd.printf("Connect Failed\n");
    reset();
    return;
  }

  if(!is_registered)
  {
    API_register();
    delay(50);
  }
  else if(audioIndex == 0)
  {
    Serial.println("Starting Recording");
    M5.Lcd.printf("Starting Recording\n");
    size_t bytesread;

    while (audioIndex < TOTAL_SAMPLES) {
        // Read raw I2S samples into int16_t buffer
        i2s_read(I2S_NUM_0, (char *)rawBuffer, READ_LEN, &bytesread, portMAX_DELAY);
        int samples_read = bytesread / sizeof(int16_t);
        //Serial.println(samples_read);
        for (int i = 0; i < samples_read && audioIndex < TOTAL_SAMPLES; ++i) {
            // Normalize to float32 [-1.0, 1.0]
            float normalized = rawBuffer[i] / 32768.0f;
            audioBuffer[audioIndex++] = normalized;
        }
    }
    Serial.println("Recording Over");
    M5.Lcd.printf("Recording Done\n");
  }
  
}
