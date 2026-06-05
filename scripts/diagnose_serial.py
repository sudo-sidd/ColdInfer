import serial
import time
import sys

port = '/dev/ttyUSB0'
baudrates = [115200, 74880, 9600]

print("Testing serial connection on different baud rates to diagnose garbage characters...")

for baud in baudrates:
    print(f"\n--- Trying {baud} baud ---")
    try:
        ser = serial.Serial(port, baud, timeout=2)
        start_time = time.time()
        output = b""
        while time.time() - start_time < 3:
            if ser.in_waiting:
                output += ser.read(ser.in_waiting)
            time.sleep(0.1)
        
        ser.close()
        
        # Try to decode what we got
        if len(output) > 0:
            print(f"Raw hex: {output[:50].hex()}...")
            try:
                decoded = output.decode('utf-8', errors='replace')
                print(f"Decoded: {decoded.strip()}")
            except Exception as e:
                print(f"Could not decode text: {e}")
        else:
            print("No data received in 3 seconds.")
            
    except Exception as e:
        print(f"Error opening serial port at {baud}: {e}")
