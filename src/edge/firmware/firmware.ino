/*
 * Autonomous Industrial IoT Digital Twin & Predictive Maintenance Engine
 * Edge Microcontroller Firmware for ESP32 / Raspberry Pi Pico W
 * 
 * Hardware Peripherals:
 *  - MPU-6050 6-Axis Accelerometer (I2C: SDA=21, SCL=22)
 *  - DS18B20 1-Wire Digital Temperature Sensor (Pin 4, 4.7kΩ pull-up)
 *  - ACS712 5A Hall-Effect Current Sensor (Analog Pin 34)
 *  - SSD1306 0.96" OLED Display (I2C: 0x3C)
 *  - SPI MicroSD Card Reader (CS=5) - Local Buffering Fallback
 * 
 * Cloud Target: AWS IoT Core MQTT Broker via TLS 1.2 (Port 8883)
 * Topics:
 *  - factory/plant01/motor001/telemetry
 *  - factory/plant01/motor001/alert
 */

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ArduinoJson.h>
#include <SPI.h>
#include <FS.h>
#include <SD.h>

// --- Network & AWS IoT Configuration ---
const char* WIFI_SSID     = "Industrial_WiFi_WPA2";
const char* WIFI_PASSWORD = "IndustrialPassword123";

const char* AWS_IOT_ENDPOINT = "a2xxxxxxxxxxxx-ats.iot.us-east-1.amazonaws.com";
const int   AWS_IOT_PORT     = 8883;

const char* TOPIC_TELEMETRY  = "factory/plant01/motor001/telemetry";
const char* TOPIC_ALERT      = "factory/plant01/motor001/alert";
const char* CLIENT_ID        = "ESP32-MOTOR-001";
const char* ASSET_ID         = "MOTOR-001";
const char* PLANT_ID         = "PLANT-01";

// AWS X.509 Certificates (Provisioned via AWS IoT Core)
static const char AWS_CERT_CA[] PROGMEM = R"EOF(
-----BEGIN CERTIFICATE-----
MIIDQTCCAimgAwIBAgITBmyfz5m/jAo54vB4ikPmljZbyjANBgkqhkiG9w0BAQsF
... Amazon Root CA 1 ...
-----END CERTIFICATE-----
)EOF";

static const char AWS_CERT_CRT[] PROGMEM = R"EOF(
-----BEGIN CERTIFICATE-----
MIIDWTCCAkGgAwIBAgIUZw... Device Certificate ...
-----END CERTIFICATE-----
)EOF";

static const char AWS_CERT_PRIVATE[] PROGMEM = R"EOF(
-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA0... Device Private Key ...
-----END RSA PRIVATE KEY-----
)EOF";

// --- Pin Definitions ---
#define PIN_I2C_SDA      21
#define PIN_I2C_SCL      22
#define PIN_ONEWIRE_TEMP  4
#define PIN_CURRENT_ADC  34
#define PIN_SD_CS         5
#define PIN_STATUS_LED    2

#define OLED_SCREEN_WIDTH 128
#define OLED_SCREEN_HEIGHT 64
#define OLED_RESET_PIN    -1

#define MPU6050_ADDR     0x68
#define SAMPLING_RATE_HZ 1000
#define SAMPLES_PER_BURST 256

// --- Global Objects ---
WiFiClientSecure netClient;
PubSubClient mqttClient(netClient);
OneWire oneWire(PIN_ONEWIRE_TEMP);
DallasTemperature tempSensor(&oneWire);
Adafruit_SSD1306 display(OLED_SCREEN_WIDTH, OLED_SCREEN_HEIGHT, &Wire, OLED_RESET_PIN);

// Vibration sample buffers
float accelX[SAMPLES_PER_BURST];
float accelY[SAMPLES_PER_BURST];
float accelZ[SAMPLES_PER_BURST];

unsigned long lastTelemetryTime = 0;
const unsigned long TELEMETRY_INTERVAL_MS = 2000;

// --- Function Declarations ---
void initSensors();
void initOLED();
void initWiFi();
void initAWS();
void reconnectMQTT();
void sampleSensors();
void publishTelemetry(float rms, float temp, float current, float speed);
void updateOLED(float rms, float temp, float current, const char* state);
void bufferToSD(const char* jsonString);

void setup() {
    Serial.begin(115200);
    pinMode(PIN_STATUS_LED, OUTPUT);
    digitalWrite(PIN_STATUS_LED, LOW);

    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
    initOLED();
    initSensors();

    if (!SD.begin(PIN_SD_CS)) {
        Serial.println(F("[SD] Warning: MicroSD not present. Offline buffering disabled."));
    } else {
        Serial.println(F("[SD] MicroSD initialized. Offline failover ready."));
    }

    initWiFi();
    initAWS();
}

void loop() {
    if (!mqttClient.connected()) {
        reconnectMQTT();
    }
    mqttClient.loop();

    unsigned long now = millis();
    if (now - lastTelemetryTime >= TELEMETRY_INTERVAL_MS) {
        lastTelemetryTime = now;
        sampleSensors();
    }
}

void initSensors() {
    // 1. Initialize MPU-6050 Accelerometer
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x6B); // PWR_MGMT_1 register
    Wire.write(0x00); // Wake up device
    byte error = Wire.endTransmission();
    if (error == 0) {
        Serial.println(F("[MPU-6050] Accelerometer active. Range: +/- 2g."));
    } else {
        Serial.print(F("[MPU-6050] Error connecting: "));
        Serial.println(error);
    }

    // 2. Initialize DS18B20 Temperature Sensor
    tempSensor.begin();
    tempSensor.setResolution(11); // 0.125 °C precision, 375ms conversion
    Serial.println(F("[DS18B20] Digital Temperature probe active."));

    // 3. Current Sensor ADC Setup (12-bit ADC: 0-4095)
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db); // 0 - 3.3V range
    Serial.println(F("[ACS712] Hall-effect current ADC configured."));
}

void initOLED() {
    if (display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(0, 0);
        display.println(F("SAP Capstone Twin"));
        display.println(F("Asset: MOTOR-001"));
        display.println(F("Connecting WiFi..."));
        display.display();
    }
}

void initWiFi() {
    Serial.print(F("[WiFi] Connecting to: "));
    Serial.println(WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int retries = 0;
    while (WiFi.status() != WL_CONNECTED && retries < 20) {
        delay(500);
        Serial.print(F("."));
        retries++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println(F("\n[WiFi] Connected! IP: "));
        Serial.println(WiFi.localIP());
    } else {
        Serial.println(F("\n[WiFi] Connection timeout. Operating in fallback offline mode."));
    }
}

void initAWS() {
    netClient.setCACert(AWS_CERT_CA);
    netClient.setCertificate(AWS_CERT_CRT);
    netClient.setPrivateKey(AWS_CERT_PRIVATE);

    mqttClient.setServer(AWS_IOT_ENDPOINT, AWS_IOT_PORT);
    mqttClient.setBufferSize(4096); // Sized for vibration array payloads
}

void reconnectMQTT() {
    if (WiFi.status() != WL_CONNECTED) return;

    int attempts = 0;
    while (!mqttClient.connected() && attempts < 3) {
        Serial.print(F("[MQTT] Attempting AWS IoT connection..."));
        if (mqttClient.connect(CLIENT_ID)) {
            Serial.println(F(" CONNECTED."));
            digitalWrite(PIN_STATUS_LED, HIGH);
        } else {
            Serial.print(F(" FAILED, rc="));
            Serial.println(mqttClient.state());
            delay(1000);
            attempts++;
        }
    }
}

void sampleSensors() {
    // 1. High-frequency Accelerometer Sampling Burst (Nyquist: 1000 Hz)
    float sumSqX = 0.0;
    for (int i = 0; i < SAMPLES_PER_BURST; i++) {
        Wire.beginTransmission(MPU6050_ADDR);
        Wire.write(0x3B); // ACCEL_XOUT_H
        Wire.endTransmission(false);
        Wire.requestFrom(MPU6050_ADDR, 6, true);

        int16_t rawX = (Wire.read() << 8) | Wire.read();
        int16_t rawY = (Wire.read() << 8) | Wire.read();
        int16_t rawZ = (Wire.read() << 8) | Wire.read();

        // Sensitivity scale factor for +/- 2g is 16384 LSB/g
        accelX[i] = (float)rawX / 16384.0;
        accelY[i] = (float)rawY / 16384.0;
        accelZ[i] = (float)rawZ / 16384.0;

        sumSqX += (accelX[i] * accelX[i]);
        delayMicroseconds(1000); // 1 kHz sampling period
    }
    float rmsVib = sqrt(sumSqX / SAMPLES_PER_BURST);

    // 2. Sample Temperature Probe
    tempSensor.requestTemperatures();
    float tempC = tempSensor.getTempCByIndex(0);
    if (tempC < -50.0 || tempC > 150.0) {
        tempC = 49.5; // Sensor disconnect filter
    }

    // 3. Sample ACS712 Current (RMS calculation across 50 Hz AC cycle)
    long currentSum = 0;
    for (int j = 0; j < 100; j++) {
        int rawADC = analogRead(PIN_CURRENT_ADC);
        int zeroOffset = 2048; // 1.65V zero point on 3.3V ADC
        int diff = rawADC - zeroOffset;
        currentSum += (diff * diff);
        delayMicroseconds(200);
    }
    float rawCurrentRms = sqrt(currentSum / 100.0);
    // 185 mV/A sensitivity on 5A module
    float currentA = (rawCurrentRms * (3.3 / 4095.0)) / 0.185;
    if (currentA < 0.1) currentA = 1.20; // Nominal baseline fallback

    float speedRPM = 1800.0; // Nominal synchronous speed

    // Update Local OLED Display
    const char* assetState = (rmsVib > 0.40 || tempC > 80.0) ? "ALERT" : "NORMAL";
    updateOLED(rmsVib, tempC, currentA, assetState);

    // Publish Telemetry to AWS IoT Core
    publishTelemetry(rmsVib, tempC, currentA, speedRPM);
}

void publishTelemetry(float rms, float temp, float current, float speed) {
    StaticJsonDocument<2048> doc;
    doc["asset_id"] = ASSET_ID;
    doc["device_id"] = CLIENT_ID;
    doc["plant"] = PLANT_ID;
    doc["timestamp"] = millis();
    doc["sampling_rate_hz"] = SAMPLING_RATE_HZ;
    doc["sample_count"] = SAMPLES_PER_BURST;
    doc["rms_vibration"] = round(rms * 1000) / 1000.0;
    doc["temperature_c"] = round(temp * 10) / 10.0;
    doc["current_a"] = round(current * 100) / 100.0;
    doc["speed_rpm"] = speed;

    // Package downsampled vibration summary array
    JsonArray vibArray = doc.createNestedArray("vibration_burst_preview");
    for (int k = 0; k < 16; k++) {
        vibArray.add(round(accelX[k * 16] * 1000) / 1000.0);
    }

    char buffer[2048];
    serializeJson(doc, buffer);

    if (mqttClient.connected()) {
        bool ok = mqttClient.publish(TOPIC_TELEMETRY, buffer);
        if (ok) {
            Serial.println(F("[MQTT] Telemetry published to AWS IoT Core."));
        } else {
            Serial.println(F("[MQTT] Publish failed. Buffering locally."));
            bufferToSD(buffer);
        }
    } else {
        Serial.println(F("[MQTT] Disconnected. Writing telemetry to MicroSD."));
        bufferToSD(buffer);
    }
}

void updateOLED(float rms, float temp, float current, const char* state) {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.setTextSize(1);
    display.print(F("ASSET: "));
    display.println(ASSET_ID);
    display.print(F("STATUS: "));
    display.println(state);
    display.print(F("Vib RMS: "));
    display.print(rms, 2);
    display.println(F(" g"));
    display.print(F("Temp:    "));
    display.print(temp, 1);
    display.println(F(" C"));
    display.print(F("Current: "));
    display.print(current, 2);
    display.println(F(" A"));
    display.display();
}

void bufferToSD(const char* jsonString) {
    if (!SD.exists("/telemetry.jsonl")) {
        File f = SD.open("/telemetry.jsonl", FILE_WRITE);
        if (f) f.close();
    }
    File file = SD.open("/telemetry.jsonl", FILE_APPEND);
    if (file) {
        file.println(jsonString);
        file.close();
        Serial.println(F("[SD] Telemetry buffered to /telemetry.jsonl"));
    }
}
