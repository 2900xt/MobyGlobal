#include <M5StickC.h>
#include <driver/i2s.h>

#define PIN_CLK     0
#define PIN_DATA    34
#define SAMPLE_RATE 8000
#define DURATION_SEC 3
#define CHANNELS    1

// Total number of samples to collect
#define TOTAL_SAMPLES (SAMPLE_RATE * DURATION_SEC)
#define READ_LEN      512

int16_t rawBuffer[READ_LEN / 2];
float audioBuffer[TOTAL_SAMPLES];
int audioIndex = 0;

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

void mic_record_task(void *arg) {
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

    // Optionally signal with LED or vibration or loopback
    vTaskDelete(NULL);  // Stop task when done
}

void setup() {
    M5.begin();
    Serial.begin(115200);
    delay(1000);
    
    M5.Lcd.setRotation(0);
    M5.Lcd.fillScreen(BLACK);
    M5.Lcd.setTextColor(WHITE);

    i2sInit();
    xTaskCreate(mic_record_task, "mic_record_task", 4096, NULL, 1, NULL);
}

void loop() {
    // Idle loop
    vTaskDelay(1000 / portTICK_RATE_MS);
}
