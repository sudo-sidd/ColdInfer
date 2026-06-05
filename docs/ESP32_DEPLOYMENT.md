# ESP32 Deployment Guide

ColdInfer supports bare-metal inference on ESP32 microcontrollers. This guide explains how to flash the compiled engine and model to an ESP32, monitor the output, and test the inference API.

## Requirements

Ensure you have the required dependencies installed. If you are using the project's virtual environment, everything should be set up automatically via `uv sync`.

```bash
uv sync
```

Dependencies include `esptool`, `pyserial`, and `requests` (for testing the API).

## 1. Flashing the Model to ESP32

To flash the compiled bootloader, partition table, and the ColdInfer ESP32 firmware to the device, use `esptool`. Replace `/dev/ttyUSB1` with the appropriate serial port for your ESP32.

```bash
sudo $(which esptool) --port /dev/ttyUSB1 --chip esp32 -b 460800 write-flash \
  --flash-mode dio \
  --flash-size 4MB \
  --flash-freq 40m \
  0x1000 cpp/esp32_inference/build/bootloader/bootloader.bin \
  0x8000 cpp/esp32_inference/build/partition_table/partition-table.bin \
  0x10000 cpp/esp32_inference/build/esp32_inference.bin
```

## 2. Monitoring the Serial Output

After flashing, you can monitor the ESP32's serial output to see boot messages, IP address assignment (if connected to Wi-Fi), and server initialization.

```bash
sudo screen /dev/ttyUSB1 115200
```
*(To exit `screen`, press `Ctrl+A`, then `K`, and confirm with `y`.)*

## 3. Testing the Inference API

Once the ESP32 is running and connected to the network, it will host an HTTP server that accepts inference requests. Use the provided test script to send a generated tensor to the ESP32 and measure the inference results and latency.

Replace `10.205.64.92` with the actual IP address of your ESP32.

```bash
uv run scripts/test_esp_api.py 10.205.64.92
```

### Example Output

```text
Generating 784-element float array...
Sending POST request to http://10.205.64.92/infer...

--- ESP32 Inference Results ---
Latency:      76671 us
Free Memory:  215 KB
Probabilities:
  Class 0: 0.1045
  Class 1: 0.1662
  Class 2: 0.1360
  Class 3: 0.1135
  Class 4: 0.0407
  Class 5: 0.0565
  Class 6: 0.0749
  Class 7: 0.1331
  Class 8: 0.1139
  Class 9: 0.0606
```
